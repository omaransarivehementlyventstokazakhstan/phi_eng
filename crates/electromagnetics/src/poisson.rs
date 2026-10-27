//! The electrostatic grid: a red-black SOR solve of the Poisson equation.
//!
//! The solve is `nabla^2 phi = -rho / (eps0 * eps_r)` on a uniform grid.
//! The electric field is `E = -grad(phi)`. Sources are deposited with
//! bilinear/trilinear weights (cloud-in-cell), so a point charge spreads
//! over the cells it sits between instead of landing on one node.
//!
//! # 2D and 3D are different physical models
//!
//! The pairwise solver in [`crate::electrostatics`] is the **3D point
//! charge law** (`1/r^2`) evaluated in the plane. The 2D Poisson
//! equation has the **2D line-charge law** (`1/r`, logarithmic potential),
//! because the Laplacian in two dimensions has a different Green's
//! function. The two are therefore not interchangeable; the 2D field mode
//! is a 2D electrostatic model and the pairwise mode is the 3D law. This
//! is recorded as a known limitation with a pinned test. The 3D grid and
//! the 3D pairwise solver agree.

use glam::{Vec2, Vec3};

use crate::constants::{eps0, MAX_FIELD_CELLS};
use crate::state::{ElectroError, FieldBoundary, FieldConfig2, FieldConfig3};

/// A 2D electrostatic grid.
#[derive(Debug, Clone)]
pub struct PoissonGrid2 {
    cfg: FieldConfig2,
    coulomb_k: f32,
    phi: Vec<f32>,
    rho: Vec<f32>,
}

impl PoissonGrid2 {
    /// Build a grid from its configuration and the Coulomb constant.
    ///
    /// The constant fixes `eps0 = 1 / (4 pi k)`, so the solved field is
    /// already in the same units as the pairwise force.
    pub fn new(cfg: FieldConfig2, coulomb_k: f32) -> Result<Self, ElectroError> {
        if cfg.nx == 0 || cfg.ny == 0 {
            return Err(ElectroError::InvalidGrid);
        }
        if !cfg.dx.is_finite() || cfg.dx <= 0.0 || !cfg.origin.is_finite() {
            return Err(ElectroError::InvalidGrid);
        }
        let cells = cfg.nx as usize * cfg.ny as usize;
        if cells > MAX_FIELD_CELLS {
            return Err(ElectroError::GridTooLarge);
        }
        if cfg.iterations == 0 || !cfg.omega.is_finite() || cfg.omega <= 0.0 || cfg.omega >= 2.0 {
            return Err(ElectroError::InvalidGrid);
        }
        if !cfg.eps_r.is_finite() || cfg.eps_r < crate::constants::MIN_MATERIAL {
            return Err(ElectroError::InvalidPermittivity);
        }
        if !coulomb_k.is_finite() || coulomb_k <= 0.0 {
            return Err(ElectroError::NonFinite);
        }
        Ok(Self {
            cfg,
            coulomb_k,
            phi: vec![0.0; cells],
            rho: vec![0.0; cells],
        })
    }

    /// The configuration the grid was built from.
    pub fn config(&self) -> &FieldConfig2 {
        &self.cfg
    }

    /// The number of cells.
    pub fn cells(&self) -> usize {
        self.phi.len()
    }

    /// The solved potential, row-major.
    pub fn phi(&self) -> &[f32] {
        &self.phi
    }

    /// The charge density, row-major.
    pub fn rho(&self) -> &[f32] {
        &self.rho
    }

    /// Replace the potential (used by the snapshot loader).
    pub fn set_phi(&mut self, phi: Vec<f32>) -> Result<(), ElectroError> {
        if phi.len() != self.phi.len() {
            return Err(ElectroError::InvalidGrid);
        }
        self.phi = phi;
        Ok(())
    }

    /// Clear the deposited sources.
    pub fn clear_sources(&mut self) {
        for v in self.rho.iter_mut() {
            *v = 0.0;
        }
    }

    /// Deposit a point charge with bilinear weights.
    ///
    /// The charge is spread over the four cells around the point, so the
    /// total deposited charge equals `charge` exactly (to float rounding).
    /// The density is `charge / dx^2` in the 2D model.
    pub fn deposit_charge(&mut self, pos: Vec2, charge: f32) {
        if !pos.is_finite() || !charge.is_finite() || charge == 0.0 {
            return;
        }
        let nx = self.cfg.nx as i64;
        let ny = self.cfg.ny as i64;
        let fx = (pos.x - self.cfg.origin.x) / self.cfg.dx;
        let fy = (pos.y - self.cfg.origin.y) / self.cfg.dx;
        if !fx.is_finite() || !fy.is_finite() {
            return;
        }
        let ix = fx.floor();
        let iy = fy.floor();
        let tx = fx - ix;
        let ty = fy - iy;
        let i0 = ix as i64;
        let j0 = iy as i64;
        let density = charge / (self.cfg.dx * self.cfg.dx);
        let w = [
            (i0, j0, (1.0 - tx) * (1.0 - ty)),
            (i0 + 1, j0, tx * (1.0 - ty)),
            (i0, j0 + 1, (1.0 - tx) * ty),
            (i0 + 1, j0 + 1, tx * ty),
        ];
        for (i, j, w) in w {
            if w == 0.0 {
                continue;
            }
            let ii = wrap_or_clamp(i, nx, self.cfg.boundary);
            let jj = wrap_or_clamp(j, ny, self.cfg.boundary);
            let idx = (jj * nx + ii) as usize;
            self.rho[idx] += density * w;
        }
    }

    /// Solve the Poisson equation with red-black SOR.
    ///
    /// The iteration order and count are fixed, so reruns are
    /// bit-identical. `rho` and `phi` are both interpreted per cell.
    pub fn solve(&mut self) {
        let nx = self.cfg.nx as usize;
        let ny = self.cfg.ny as usize;
        let dx2 = self.cfg.dx * self.cfg.dx;
        let scale = dx2 / (eps0(self.coulomb_k) * self.cfg.eps_r);
        for _ in 0..self.cfg.iterations {
            for color in 0..2usize {
                for j in 0..ny {
                    for i in 0..nx {
                        if (i + j) % 2 != color {
                            continue;
                        }
                        let idx = j * nx + i;
                        let left = self.phi_at(i as i64 - 1, j as i64);
                        let right = self.phi_at(i as i64 + 1, j as i64);
                        let down = self.phi_at(i as i64, j as i64 - 1);
                        let up = self.phi_at(i as i64, j as i64 + 1);
                        let target = (left + right + down + up + scale * self.rho[idx]) * 0.25;
                        let old = self.phi[idx];
                        self.phi[idx] = old + self.cfg.omega * (target - old);
                    }
                }
            }
        }
    }

    /// The potential at a world point, bilinearly interpolated.
    pub fn potential_at(&self, pos: Vec2) -> f32 {
        let (i0, j0, tx, ty) = self.locate(pos);
        let p00 = self.cell_phi(i0, j0);
        let p10 = self.cell_phi(i0 + 1, j0);
        let p01 = self.cell_phi(i0, j0 + 1);
        let p11 = self.cell_phi(i0 + 1, j0 + 1);
        let a = p00 + (p10 - p00) * tx;
        let b = p01 + (p11 - p01) * tx;
        a + (b - a) * ty
    }

    /// The electric field at a world point: `-grad(phi)`, bilinearly
    /// interpolated from the cell-centred differences.
    pub fn field_at(&self, pos: Vec2) -> Vec2 {
        let (i0, j0, tx, ty) = self.locate(pos);
        let e00 = self.cell_field(i0, j0);
        let e10 = self.cell_field(i0 + 1, j0);
        let e01 = self.cell_field(i0, j0 + 1);
        let e11 = self.cell_field(i0 + 1, j0 + 1);
        let a = e00 + (e10 - e00) * tx;
        let b = e01 + (e11 - e01) * tx;
        a + (b - a) * ty
    }

    /// The electrostatic energy `0.5 * sum(rho * phi) * dx^2`.
    pub fn energy(&self) -> f32 {
        let mut sum = 0.0f32;
        for (r, p) in self.rho.iter().zip(self.phi.iter()) {
            sum += r * p;
        }
        0.5 * sum * self.cfg.dx * self.cfg.dx
    }

    fn locate(&self, pos: Vec2) -> (i64, i64, f32, f32) {
        let nx = self.cfg.nx as i64;
        let ny = self.cfg.ny as i64;
        let fx = (pos.x - self.cfg.origin.x) / self.cfg.dx;
        let fy = (pos.y - self.cfg.origin.y) / self.cfg.dx;
        if !fx.is_finite() || !fy.is_finite() {
            return (0, 0, 0.0, 0.0);
        }
        let cx = fx.clamp(0.0, (nx - 1) as f32);
        let cy = fy.clamp(0.0, (ny - 1) as f32);
        let ix = cx.floor() as i64;
        let iy = cy.floor() as i64;
        let tx = (cx - ix as f32).clamp(0.0, 1.0);
        let ty = (cy - iy as f32).clamp(0.0, 1.0);
        (ix, iy, tx, ty)
    }

    fn index(&self, i: i64, j: i64) -> Option<usize> {
        let nx = self.cfg.nx as i64;
        let ny = self.cfg.ny as i64;
        if i < 0 || j < 0 || i >= nx || j >= ny {
            None
        } else {
            Some((j * nx + i) as usize)
        }
    }

    /// The potential at a cell, honouring the boundary condition.
    fn phi_at(&self, i: i64, j: i64) -> f32 {
        match self.index(i, j) {
            Some(idx) => self.phi[idx],
            None => match self.cfg.boundary {
                FieldBoundary::Dirichlet => 0.0,
                FieldBoundary::Neumann => {
                    let ii = i.clamp(0, self.cfg.nx as i64 - 1);
                    let jj = j.clamp(0, self.cfg.ny as i64 - 1);
                    self.phi[(jj * self.cfg.nx as i64 + ii) as usize]
                }
                FieldBoundary::Periodic => {
                    let ii = i.rem_euclid(self.cfg.nx as i64);
                    let jj = j.rem_euclid(self.cfg.ny as i64);
                    self.phi[(jj * self.cfg.nx as i64 + ii) as usize]
                }
            },
        }
    }

    fn cell_phi(&self, i: i64, j: i64) -> f32 {
        let ii = i.clamp(0, self.cfg.nx as i64 - 1);
        let jj = j.clamp(0, self.cfg.ny as i64 - 1);
        self.phi[(jj * self.cfg.nx as i64 + ii) as usize]
    }

    fn cell_field(&self, i: i64, j: i64) -> Vec2 {
        let ex = -(self.phi_at(i + 1, j) - self.phi_at(i - 1, j)) / (2.0 * self.cfg.dx);
        let ey = -(self.phi_at(i, j + 1) - self.phi_at(i, j - 1)) / (2.0 * self.cfg.dx);
        Vec2::new(ex, ey)
    }
}

/// A 3D electrostatic grid. See [`PoissonGrid2`] for the model notes;
/// the 3D grid solves the same equation as the 3D point-charge law.
#[derive(Debug, Clone)]
pub struct PoissonGrid3 {
    cfg: FieldConfig3,
    coulomb_k: f32,
    phi: Vec<f32>,
    rho: Vec<f32>,
}

impl PoissonGrid3 {
    /// Build a grid from its configuration and the Coulomb constant.
    pub fn new(cfg: FieldConfig3, coulomb_k: f32) -> Result<Self, ElectroError> {
        if cfg.nx == 0 || cfg.ny == 0 || cfg.nz == 0 {
            return Err(ElectroError::InvalidGrid);
        }
        if !cfg.dx.is_finite() || cfg.dx <= 0.0 || !cfg.origin.is_finite() {
            return Err(ElectroError::InvalidGrid);
        }
        let cells = cfg.nx as usize * cfg.ny as usize * cfg.nz as usize;
        if cells > MAX_FIELD_CELLS {
            return Err(ElectroError::GridTooLarge);
        }
        if cfg.iterations == 0 || !cfg.omega.is_finite() || cfg.omega <= 0.0 || cfg.omega >= 2.0 {
            return Err(ElectroError::InvalidGrid);
        }
        if !cfg.eps_r.is_finite() || cfg.eps_r < crate::constants::MIN_MATERIAL {
            return Err(ElectroError::InvalidPermittivity);
        }
        if !coulomb_k.is_finite() || coulomb_k <= 0.0 {
            return Err(ElectroError::NonFinite);
        }
        Ok(Self {
            cfg,
            coulomb_k,
            phi: vec![0.0; cells],
            rho: vec![0.0; cells],
        })
    }

    /// The configuration.
    pub fn config(&self) -> &FieldConfig3 {
        &self.cfg
    }

    /// The number of cells.
    pub fn cells(&self) -> usize {
        self.phi.len()
    }

    /// The solved potential, row-major `(z, y, x)`.
    pub fn phi(&self) -> &[f32] {
        &self.phi
    }

    /// The charge density, row-major `(z, y, x)`.
    pub fn rho(&self) -> &[f32] {
        &self.rho
    }

    /// Replace the potential (snapshot loader).
    pub fn set_phi(&mut self, phi: Vec<f32>) -> Result<(), ElectroError> {
        if phi.len() != self.phi.len() {
            return Err(ElectroError::InvalidGrid);
        }
        self.phi = phi;
        Ok(())
    }

    /// Clear the deposited sources.
    pub fn clear_sources(&mut self) {
        for v in self.rho.iter_mut() {
            *v = 0.0;
        }
    }

    /// Deposit a point charge with trilinear weights.
    pub fn deposit_charge(&mut self, pos: Vec3, charge: f32) {
        if !pos.is_finite() || !charge.is_finite() || charge == 0.0 {
            return;
        }
        let fx = (pos.x - self.cfg.origin.x) / self.cfg.dx;
        let fy = (pos.y - self.cfg.origin.y) / self.cfg.dx;
        let fz = (pos.z - self.cfg.origin.z) / self.cfg.dx;
        if !fx.is_finite() || !fy.is_finite() || !fz.is_finite() {
            return;
        }
        let ix = fx.floor();
        let iy = fy.floor();
        let iz = fz.floor();
        let tx = fx - ix;
        let ty = fy - iy;
        let tz = fz - iz;
        let i0 = ix as i64;
        let j0 = iy as i64;
        let k0 = iz as i64;
        let density = charge / (self.cfg.dx * self.cfg.dx * self.cfg.dx);
        let nx = self.cfg.nx as i64;
        let ny = self.cfg.ny as i64;
        let nz = self.cfg.nz as i64;
        for (dz, wz) in [(0, 1.0 - tz), (1, tz)] {
            for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
                for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
                    let w = wx * wy * wz;
                    if w == 0.0 {
                        continue;
                    }
                    let i = wrap_or_clamp(i0 + dx, nx, self.cfg.boundary);
                    let j = wrap_or_clamp(j0 + dy, ny, self.cfg.boundary);
                    let k = wrap_or_clamp(k0 + dz, nz, self.cfg.boundary);
                    let idx = ((k * ny + j) * nx + i) as usize;
                    self.rho[idx] += density * w;
                }
            }
        }
    }

    /// Solve with red-black SOR.
    pub fn solve(&mut self) {
        let nx = self.cfg.nx as usize;
        let ny = self.cfg.ny as usize;
        let nz = self.cfg.nz as usize;
        let dx2 = self.cfg.dx * self.cfg.dx;
        let scale = dx2 / (eps0(self.coulomb_k) * self.cfg.eps_r);
        for _ in 0..self.cfg.iterations {
            for color in 0..2usize {
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            if (i + j + k) % 2 != color {
                                continue;
                            }
                            let idx = (k * ny + j) * nx + i;
                            let left = self.phi_at(i as i64 - 1, j as i64, k as i64);
                            let right = self.phi_at(i as i64 + 1, j as i64, k as i64);
                            let down = self.phi_at(i as i64, j as i64 - 1, k as i64);
                            let up = self.phi_at(i as i64, j as i64 + 1, k as i64);
                            let back = self.phi_at(i as i64, j as i64, k as i64 - 1);
                            let front = self.phi_at(i as i64, j as i64, k as i64 + 1);
                            let target =
                                (left + right + down + up + back + front + scale * self.rho[idx])
                                    / 6.0;
                            let old = self.phi[idx];
                            self.phi[idx] = old + self.cfg.omega * (target - old);
                        }
                    }
                }
            }
        }
    }

    /// The potential at a world point, trilinearly interpolated.
    pub fn potential_at(&self, pos: Vec3) -> f32 {
        let (i0, j0, k0, tx, ty, tz) = self.locate(pos);
        let mut acc = 0.0f32;
        for (dz, wz) in [(0, 1.0 - tz), (1, tz)] {
            for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
                for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
                    let w = wx * wy * wz;
                    if w == 0.0 {
                        continue;
                    }
                    acc += w * self.cell_phi(i0 + dx, j0 + dy, k0 + dz);
                }
            }
        }
        acc
    }

    /// The electric field at a world point: `-grad(phi)`, trilinearly
    /// interpolated from the cell-centred differences.
    pub fn field_at(&self, pos: Vec3) -> Vec3 {
        let (i0, j0, k0, tx, ty, tz) = self.locate(pos);
        let mut acc = Vec3::ZERO;
        for (dz, wz) in [(0, 1.0 - tz), (1, tz)] {
            for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
                for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
                    let w = wx * wy * wz;
                    if w == 0.0 {
                        continue;
                    }
                    acc += w * self.cell_field(i0 + dx, j0 + dy, k0 + dz);
                }
            }
        }
        acc
    }

    /// The electrostatic energy `0.5 * sum(rho * phi) * dx^3`.
    pub fn energy(&self) -> f32 {
        let mut sum = 0.0f32;
        for (r, p) in self.rho.iter().zip(self.phi.iter()) {
            sum += r * p;
        }
        0.5 * sum * self.cfg.dx * self.cfg.dx * self.cfg.dx
    }

    fn locate(&self, pos: Vec3) -> (i64, i64, i64, f32, f32, f32) {
        let nx = self.cfg.nx as i64;
        let ny = self.cfg.ny as i64;
        let nz = self.cfg.nz as i64;
        let fx = (pos.x - self.cfg.origin.x) / self.cfg.dx;
        let fy = (pos.y - self.cfg.origin.y) / self.cfg.dx;
        let fz = (pos.z - self.cfg.origin.z) / self.cfg.dx;
        if !fx.is_finite() || !fy.is_finite() || !fz.is_finite() {
            return (0, 0, 0, 0.0, 0.0, 0.0);
        }
        let cx = fx.clamp(0.0, (nx - 1) as f32);
        let cy = fy.clamp(0.0, (ny - 1) as f32);
        let cz = fz.clamp(0.0, (nz - 1) as f32);
        let ix = cx.floor() as i64;
        let iy = cy.floor() as i64;
        let iz = cz.floor() as i64;
        let tx = (cx - ix as f32).clamp(0.0, 1.0);
        let ty = (cy - iy as f32).clamp(0.0, 1.0);
        let tz = (cz - iz as f32).clamp(0.0, 1.0);
        (ix, iy, iz, tx, ty, tz)
    }

    fn index(&self, i: i64, j: i64, k: i64) -> Option<usize> {
        let nx = self.cfg.nx as i64;
        let ny = self.cfg.ny as i64;
        let nz = self.cfg.nz as i64;
        if i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz {
            None
        } else {
            Some(((k * ny + j) * nx + i) as usize)
        }
    }

    fn phi_at(&self, i: i64, j: i64, k: i64) -> f32 {
        match self.index(i, j, k) {
            Some(idx) => self.phi[idx],
            None => match self.cfg.boundary {
                FieldBoundary::Dirichlet => 0.0,
                FieldBoundary::Neumann => {
                    let ii = i.clamp(0, self.cfg.nx as i64 - 1);
                    let jj = j.clamp(0, self.cfg.ny as i64 - 1);
                    let kk = k.clamp(0, self.cfg.nz as i64 - 1);
                    self.phi[((kk * self.cfg.ny as i64 + jj) * self.cfg.nx as i64 + ii) as usize]
                }
                FieldBoundary::Periodic => {
                    let ii = i.rem_euclid(self.cfg.nx as i64);
                    let jj = j.rem_euclid(self.cfg.ny as i64);
                    let kk = k.rem_euclid(self.cfg.nz as i64);
                    self.phi[((kk * self.cfg.ny as i64 + jj) * self.cfg.nx as i64 + ii) as usize]
                }
            },
        }
    }

    fn cell_phi(&self, i: i64, j: i64, k: i64) -> f32 {
        let ii = i.clamp(0, self.cfg.nx as i64 - 1);
        let jj = j.clamp(0, self.cfg.ny as i64 - 1);
        let kk = k.clamp(0, self.cfg.nz as i64 - 1);
        self.phi[((kk * self.cfg.ny as i64 + jj) * self.cfg.nx as i64 + ii) as usize]
    }

    fn cell_field(&self, i: i64, j: i64, k: i64) -> Vec3 {
        let ex = -(self.phi_at(i + 1, j, k) - self.phi_at(i - 1, j, k)) / (2.0 * self.cfg.dx);
        let ey = -(self.phi_at(i, j + 1, k) - self.phi_at(i, j - 1, k)) / (2.0 * self.cfg.dx);
        let ez = -(self.phi_at(i, j, k + 1) - self.phi_at(i, j, k - 1)) / (2.0 * self.cfg.dx);
        Vec3::new(ex, ey, ez)
    }
}

fn wrap_or_clamp(i: i64, n: i64, boundary: FieldBoundary) -> i64 {
    match boundary {
        FieldBoundary::Periodic => i.rem_euclid(n),
        _ => i.clamp(0, n - 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg2() -> FieldConfig2 {
        FieldConfig2 {
            nx: 64,
            ny: 64,
            dx: 0.1,
            origin: Vec2::splat(-3.2),
            iterations: 200,
            omega: 1.9,
            boundary: FieldBoundary::Dirichlet,
            eps_r: 1.0,
            snapshot_field: true,
        }
    }

    #[test]
    fn a_positive_charge_makes_a_positive_potential() {
        let mut g = PoissonGrid2::new(cfg2(), 1.0).unwrap();
        g.deposit_charge(Vec2::ZERO, 1.0);
        g.solve();
        let v = g.potential_at(Vec2::new(0.5, 0.0));
        assert!(v > 0.0, "potential = {v}");
        let e = g.field_at(Vec2::new(0.5, 0.0));
        assert!(e.x > 0.0, "field = {e:?}");
    }

    #[test]
    fn opposite_charges_make_a_dipole() {
        let mut g = PoissonGrid2::new(cfg2(), 1.0).unwrap();
        g.deposit_charge(Vec2::new(-0.5, 0.0), 1.0);
        g.deposit_charge(Vec2::new(0.5, 0.0), -1.0);
        g.solve();
        let left = g.potential_at(Vec2::new(-1.0, 0.0));
        let right = g.potential_at(Vec2::new(1.0, 0.0));
        assert!(left > right, "left={left} right={right}");
    }

    #[test]
    fn deposit_conserves_the_total_charge() {
        let mut g = PoissonGrid2::new(cfg2(), 1.0).unwrap();
        g.deposit_charge(Vec2::new(0.13, -0.07), 2.5);
        let total: f32 = g.rho().iter().sum();
        let expected = 2.5 / (0.1 * 0.1);
        assert!((total - expected).abs() / expected.abs() < 1e-4);
    }

    #[test]
    fn oversized_grid_is_rejected() {
        let mut c = cfg2();
        c.nx = 4000;
        c.ny = 4000;
        assert!(matches!(
            PoissonGrid2::new(c, 1.0),
            Err(ElectroError::GridTooLarge)
        ));
    }

    fn cfg3() -> FieldConfig3 {
        FieldConfig3 {
            nx: 32,
            ny: 32,
            nz: 32,
            dx: 0.2,
            origin: Vec3::splat(-3.2),
            iterations: 200,
            omega: 1.9,
            boundary: FieldBoundary::Dirichlet,
            eps_r: 1.0,
            snapshot_field: true,
        }
    }

    #[test]
    fn three_d_charge_field_points_away() {
        let mut g = PoissonGrid3::new(cfg3(), 1.0).unwrap();
        g.deposit_charge(Vec3::ZERO, 1.0);
        g.solve();
        let e = g.field_at(Vec3::new(0.6, 0.0, 0.0));
        assert!(e.x > 0.0, "field = {e:?}");
    }

    #[test]
    fn three_d_deposit_conserves_charge() {
        let mut g = PoissonGrid3::new(cfg3(), 1.0).unwrap();
        g.deposit_charge(Vec3::new(0.13, -0.07, 0.21), 2.5);
        let total: f32 = g.rho().iter().sum();
        let expected = 2.5 / (0.2 * 0.2 * 0.2);
        assert!((total - expected).abs() / expected.abs() < 1e-4);
    }
}
