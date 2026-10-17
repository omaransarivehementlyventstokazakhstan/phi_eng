//! SPH fluid (3D): weakly-compressible particle fluid coupled to rigid bodies.
//!
//! Mirror of [`crate::fluid`] in 3D: density via the 3D poly6 kernel,
//! pressure via a clamped linear equation of state, spiky-gradient
//! pressure force, XSPH viscosity, and two-way boundary coupling through
//! [`crate::body::RigidBody3::apply_impulse`] plus pressure-sampled
//! buoyancy and Morison-style drag. Stepped as phase 1b of
//! [`crate::World3::step`]; a no-op when empty. All loops run in index
//! order with fixed neighbor-cell order, so fluid stepping is
//! deterministic.
//!
//! Boundary coverage: sphere / box / capsule shapes are analytic;
//! convex hulls and meshes use a nearest-vertex-sphere approximation
//! (documented gap: bumpy response on coarse hulls, same policy as the
//! 2D polygon impulse-only path â€” no silently wrong smooth normals).
//! Buoyancy sampling covers sphere / box / capsule; hulls and meshes
//! are impulse-coupled only.

use super::body::{RigidBody3, Shape3};
use crate::noise::{curl_3, fbm_3};
use glam::Vec3;
use phieng_math::{EPS, EPS_SQ};
use std::collections::HashMap;

/// Hard cap on stored particles per 3D fluid state (creation-time guard
/// against runaway allocation).
pub const MAX_PARTICLES_3D: usize = 4096;

/// Single SPH particle: position, velocity, and per-step density/pressure.
#[derive(Debug, Clone, Copy)]
pub struct Particle3 {
    /// World-space position.
    pub position: Vec3,
    /// World-space velocity.
    pub velocity: Vec3,
    /// Smoothed density from the poly6 sum (recomputed every step).
    pub density: f32,
    /// Pressure from the equation of state (recomputed every step).
    pub pressure: f32,
}

/// Tunables for the 3D SPH fluid. Defaults suit unit-scale scenes
/// (`rest_density = 1`, decimeter particles at 60 Hz).
#[derive(Debug, Clone, Copy)]
pub struct FluidConfig3 {
    /// Uniform particle mass. Calibrated so a settled lattice at the
    /// default spacing (0.25) and support (0.5) sums to `rest_density`
    /// (measured kernel sum â‰ˆ 16 per unit mass). Rescale proportionally
    /// if you change spacing or support radius.
    pub particle_mass: f32,
    /// Rest density of the fluid.
    pub rest_density: f32,
    /// Kernel support radius. Should cover ~2 particle spacings.
    pub smoothing_length: f32,
    /// Pressure stiffness `k` in `p = max(0, k * (rho - rho0))` (linear
    /// EOS, `gamma == 1`). In Tait mode (`gamma > 1`) this acts as the
    /// `B`-scale of `p = max(0, k*rho0*((rho/rho0)^gamma - 1))`.
    pub stiffness: f32,
    /// Equation-of-state exponent: `1.0` = legacy linear (bit-identical),
    /// `> 1` = Tait/WCSPH (classic water value `7.0`).
    pub gamma: f32,
    /// Monaghan artificial-viscosity `alpha` in `[0, 2]` (default `0`).
    pub artificial_viscosity: f32,
    /// Akinci-style surface-tension (cohesion) strength `kappa` (default `0`).
    pub surface_tension: f32,
    /// CFL factor for adaptive substepping (default `0.0` = off).
    pub cfl: f32,
    /// Maximum substeps per world step when `cfl` is enabled (default `1`).
    pub max_substeps: u32,
    /// XSPH smoothing factor in `[0, 1]` (velocity diffusion strength).
    pub viscosity: f32,
    /// Quadratic drag coefficient applied per submerged sample area.
    pub drag: f32,
    /// Hard speed cap (safety net against ejection chains).
    pub max_velocity: f32,
    /// Collision radius used against rigid-body shapes.
    pub particle_radius: f32,
    /// Master switch. With no particles the step is a no-op either way.
    pub enabled: bool,
}

impl Default for FluidConfig3 {
    fn default() -> Self {
        Self {
            particle_mass: 0.0625,
            rest_density: 1.0,
            smoothing_length: 0.5,
            stiffness: 100.0,
            gamma: 1.0,
            artificial_viscosity: 0.0,
            surface_tension: 0.0,
            cfl: 0.0,
            max_substeps: 1,
            viscosity: 0.3,
            drag: 4.0,
            max_velocity: 30.0,
            particle_radius: 0.125,
            enabled: true,
        }
    }
}

/// 3D SPH fluid state owned by [`crate::World3`].
#[derive(Debug, Clone)]
pub struct FluidState3 {
    /// Tunables (see [`FluidConfig3`]).
    pub config: FluidConfig3,
    /// Live particles in insertion order (indices are stable).
    pub particles: Vec<Particle3>,
    /// Uniform-grid neighbor index: cell -> particle indices. Rebuilt
    /// every step; cell vectors are cleared in place, with vacated
    /// cells compacted periodically.
    grid: HashMap<(i32, i32, i32), Vec<usize>>,
    /// Per-particle force scratch, resized as needed.
    forces: Vec<Vec3>,
    /// Reused (point, normal, area) surface samples for buoyancy.
    sample_scratch: Vec<(Vec3, Vec3, f32)>,
    /// Steps since vacated grid cells were compacted.
    clears_since_compact: u32,
}

impl FluidState3 {
    /// Empty fluid with default config.
    pub fn new() -> Self {
        Self {
            config: FluidConfig3::default(),
            particles: Vec::new(),
            grid: HashMap::new(),
            forces: Vec::new(),
            sample_scratch: Vec::new(),
            clears_since_compact: 0,
        }
    }

    /// Insert particles. Returns the number actually added (capped at
    /// [`MAX_PARTICLES_3D`] total; excess is dropped).
    pub fn add_particles(&mut self, positions: &[Vec3]) -> usize {
        let room = MAX_PARTICLES_3D.saturating_sub(self.particles.len());
        let n = room.min(positions.len());
        self.particles.reserve(n);
        for &p in &positions[..n] {
            self.particles.push(Particle3 {
                position: p,
                velocity: Vec3::ZERO,
                density: self.config.rest_density,
                pressure: 0.0,
            });
        }
        n
    }

    /// Remove all particles (keeps allocations).
    pub fn clear(&mut self) {
        self.particles.clear();
    }

    /// 3D poly6 kernel: `315 / (64 pi h^9) (h^2 - r^2)^3`, zero beyond `h`.
    fn poly6(r2: f32, h: f32) -> f32 {
        if r2 >= h * h || r2 < 0.0 {
            return 0.0;
        }
        let h2 = h * h;
        let t = h2 - r2;
        use std::f32::consts::PI;
        315.0 / (64.0 * PI * h2 * h2 * h2 * h2 * h) * t * t * t
    }

    /// 3D spiky gradient: `-45 / (pi h^6) (h - r)^2 * (r_vec / r)`.
    /// Zero at and beyond support and at `r == 0` (no direction).
    fn spiky_grad(r_vec: Vec3, r: f32, h: f32) -> Vec3 {
        if r >= h || r < EPS {
            return Vec3::ZERO;
        }
        use std::f32::consts::PI;
        let s = -45.0 / (PI * h * h * h * h * h * h) * (h - r) * (h - r);
        r_vec / r * s
    }

    /// Pressure from the configured equation of state (tension-free).
    /// See [`crate::fluid::FluidState2::pressure`] (private there; kept
    /// in sync). `gamma == 1` is the legacy linear form.
    fn pressure(density: f32, rest_density: f32, stiffness: f32, gamma: f32) -> f32 {
        if gamma == 1.0 {
            (stiffness * (density - rest_density)).max(0.0)
        } else {
            let rho0 = rest_density.max(EPS);
            (stiffness * rho0 * ((density / rho0).powf(gamma) - 1.0)).max(0.0)
        }
    }

    /// Monaghan artificial-viscosity pressure term (see 2D counterpart).
    fn artificial_pressure(
        delta: Vec3,
        v_rel: Vec3,
        r_sq: f32,
        h: f32,
        rho_bar: f32,
        c: f32,
        alpha: f32,
    ) -> f32 {
        let vd = v_rel.dot(delta);
        if vd >= 0.0 {
            return 0.0;
        }
        let mu = h * vd / (r_sq + 0.01 * h * h);
        -alpha * c * mu / rho_bar.max(EPS)
    }

    /// Bounded cohesion kernel (see 2D counterpart).
    fn cohesion(dist: f32, h: f32) -> f32 {
        if dist >= h {
            0.0
        } else {
            1.0 - dist / h
        }
    }

    /// Rebuild the neighbor grid from current positions. Non-finite
    /// positions are skipped (they can never map to cells).
    fn rebuild_grid(&mut self) {
        for ids in self.grid.values_mut() {
            ids.clear();
        }
        self.clears_since_compact += 1;
        if self.clears_since_compact >= 256 {
            self.clears_since_compact = 0;
            self.grid.retain(|_, ids| !ids.is_empty());
        }
        let h = self.config.smoothing_length;
        for (i, p) in self.particles.iter().enumerate() {
            if !p.position.x.is_finite() || !p.position.y.is_finite() || !p.position.z.is_finite() {
                continue;
            }
            let key = (
                (p.position.x / h).floor() as i32,
                (p.position.y / h).floor() as i32,
                (p.position.z / h).floor() as i32,
            );
            self.grid.entry(key).or_default().push(i);
        }
    }

    /// Indices of particles that may lie within the support radius of
    /// particle `i`, in deterministic cell order (includes `i` itself).
    fn neighbors(&self, i: usize, out: &mut Vec<usize>) {
        out.clear();
        let h = self.config.smoothing_length;
        let p = self.particles[i].position;
        let cx = (p.x / h).floor() as i32;
        let cy = (p.y / h).floor() as i32;
        let cz = (p.z / h).floor() as i32;
        for oz in -1..=1 {
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if let Some(ids) = self.grid.get(&(cx + ox, cy + oy, cz + oz)) {
                        for &j in ids.iter() {
                            let d2 = (self.particles[j].position - p).length_squared();
                            if d2 < h * h {
                                out.push(j);
                            }
                        }
                    }
                }
            }
        }
    }

    /// SPH pressure interpolated at a world-space point.
    /// Zero outside the fluid. Callers must have rebuilt the grid after
    /// the latest advection.
    fn sample_pressure(&self, point: Vec3) -> f32 {
        let h = self.config.smoothing_length.max(EPS);
        let m = self.config.particle_mass.max(0.0);
        let cx = (point.x / h).floor() as i32;
        let cy = (point.y / h).floor() as i32;
        let cz = (point.z / h).floor() as i32;
        let mut p = 0.0f32;
        for oz in -1..=1 {
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if let Some(ids) = self.grid.get(&(cx + ox, cy + oy, cz + oz)) {
                        for &j in ids.iter() {
                            let pj = self.particles[j];
                            let r2 = (pj.position - point).length_squared();
                            if r2 < h * h {
                                p += m * (pj.pressure / pj.density.max(EPS)) * Self::poly6(r2, h);
                            }
                        }
                    }
                }
            }
        }
        p.max(0.0)
    }

    /// Surface samples `(point, outward_normal, area)` for buoyancy:
    /// 14 sphere directions, 24 box face samples (4 per face), 18
    /// capsule samples (8 per side ring + 2 poles). Hulls and meshes
    /// are impulse-coupled only (documented gap).
    fn surface_samples(&mut self, body: &RigidBody3) {
        self.sample_scratch.clear();
        match &body.shape {
            Some(Shape3::Sphere { radius }) => {
                let dirs = [
                    Vec3::X,
                    Vec3::NEG_X,
                    Vec3::Y,
                    Vec3::NEG_Y,
                    Vec3::Z,
                    Vec3::NEG_Z,
                    Vec3::new(1.0, 1.0, 1.0).normalize_or_zero(),
                    Vec3::new(1.0, 1.0, -1.0).normalize_or_zero(),
                    Vec3::new(1.0, -1.0, 1.0).normalize_or_zero(),
                    Vec3::new(1.0, -1.0, -1.0).normalize_or_zero(),
                    Vec3::new(-1.0, 1.0, 1.0).normalize_or_zero(),
                    Vec3::new(-1.0, 1.0, -1.0).normalize_or_zero(),
                    Vec3::new(-1.0, -1.0, 1.0).normalize_or_zero(),
                    Vec3::new(-1.0, -1.0, -1.0).normalize_or_zero(),
                ];
                let area = 4.0 * std::f32::consts::PI * radius * radius / dirs.len() as f32;
                for d in dirs {
                    self.sample_scratch.push((
                        body.transform.transform_point(d * radius),
                        body.transform.rotation * d,
                        area,
                    ));
                }
            }
            Some(Shape3::Box { half_extents }) => {
                let he = *half_extents;
                let area = 8.0 * (he.x * he.y + he.y * he.z + he.z * he.x) / 24.0;
                let faces = [
                    (Vec3::Y, -he.y, 0),
                    (Vec3::NEG_Y, -he.y, 0),
                    (Vec3::X, -he.x, 1),
                    (Vec3::NEG_X, -he.x, 1),
                    (Vec3::Z, -he.z, 2),
                    (Vec3::NEG_Z, -he.z, 2),
                ];
                for (normal, fixed, axis) in faces {
                    for k in 0..4 {
                        let (t0, t1) = match k {
                            0 => (-0.75, -0.75),
                            1 => (0.75, -0.75),
                            2 => (-0.75, 0.75),
                            _ => (0.75, 0.75),
                        };
                        let local = match axis {
                            0 => Vec3::new(he.x * t0, fixed, he.z * t1),
                            1 => Vec3::new(fixed, he.y * t0, he.z * t1),
                            _ => Vec3::new(he.x * t0, he.y * t1, fixed),
                        };
                        self.sample_scratch.push((
                            body.transform.transform_point(local),
                            body.transform.rotation * normal,
                            area,
                        ));
                    }
                }
            }
            Some(Shape3::Capsule {
                radius,
                half_length,
            }) => {
                let r = *radius;
                let hl = *half_length;
                let side_area = 4.0 * std::f32::consts::PI * r * hl / 16.0;
                let cap_area = 2.0 * std::f32::consts::PI * r * r;
                for &sy in &[-hl * 0.5, hl * 0.5] {
                    for k in 0..8 {
                        let a = 2.0 * std::f32::consts::PI * k as f32 / 8.0;
                        let dir = Vec3::new(a.cos(), 0.0, a.sin());
                        self.sample_scratch.push((
                            body.transform.transform_point(dir * r + Vec3::Y * sy),
                            body.transform.rotation * dir,
                            side_area,
                        ));
                    }
                }
                for s in [-1.0f32, 1.0] {
                    self.sample_scratch.push((
                        body.transform.transform_point(Vec3::Y * s * (hl + r)),
                        body.transform.rotation * (Vec3::Y * s),
                        cap_area,
                    ));
                }
            }
            _ => {}
        }
    }

    /// Particle-vs-body boundary test. Returns the outward normal
    /// (body -> particle), penetration depth, and a contact point on
    /// the body surface. `None` when separated. Hulls and meshes use
    /// a nearest-vertex-sphere approximation (documented gap).
    fn collide_particle_body(
        &self,
        particle: Vec3,
        pr: f32,
        body: &RigidBody3,
    ) -> Option<(Vec3, f32, Vec3)> {
        match &body.shape {
            Some(Shape3::Sphere { radius }) => {
                let delta = particle - body.transform.position;
                let dist = delta.length();
                if dist >= radius + pr {
                    return None;
                }
                let normal = if dist > EPS { delta / dist } else { Vec3::X };
                let depth = radius + pr - dist;
                let point = body.transform.position + normal * radius;
                Some((normal, depth, point))
            }
            Some(Shape3::Box { half_extents }) => {
                let he = *half_extents;
                let rel = particle - body.transform.position;
                let inv = body.transform.rotation.inverse();
                let local = inv * rel;
                let clamped = Vec3::new(
                    local.x.clamp(-he.x, he.x),
                    local.y.clamp(-he.y, he.y),
                    local.z.clamp(-he.z, he.z),
                );
                let diff = local - clamped;
                if diff.length_squared() > EPS_SQ {
                    let dist = diff.length();
                    if dist >= pr {
                        return None;
                    }
                    let normal = body.transform.rotation * (diff / dist);
                    let depth = pr - dist;
                    let point = body.transform.transform_point(clamped);
                    Some((normal, depth, point))
                } else {
                    // Center inside: eject along least-penetration axis.
                    let dx = he.x - local.x.abs();
                    let dy = he.y - local.y.abs();
                    let dz = he.z - local.z.abs();
                    let (axis, s, d) = if dx <= dy && dx <= dz {
                        (Vec3::X, local.x.signum().copysign(1.0), dx)
                    } else if dy <= dz {
                        (Vec3::Y, local.y.signum().copysign(1.0), dy)
                    } else {
                        (Vec3::Z, local.z.signum().copysign(1.0), dz)
                    };
                    let normal = body.transform.rotation * (axis * s);
                    let point = body.transform.transform_point(clamped);
                    Some((normal, pr + d, point))
                }
            }
            Some(Shape3::Capsule {
                radius,
                half_length,
            }) => {
                let r = *radius;
                let hl = *half_length;
                let rel = particle - body.transform.position;
                let inv = body.transform.rotation.inverse();
                let local = inv * rel;
                // Closest point on the local Y segment [-hl, hl].
                let sy = local.y.clamp(-hl, hl);
                let seg = Vec3::new(0.0, sy, 0.0);
                let diff = local - seg;
                if diff.length_squared() > EPS_SQ {
                    let dist = diff.length();
                    if dist >= r + pr {
                        return None;
                    }
                    let normal = body.transform.rotation * (diff / dist);
                    let depth = r + pr - dist;
                    let point = body
                        .transform
                        .transform_point(seg + diff.normalize_or_zero() * r);
                    Some((normal, depth, point))
                } else {
                    // Center on the segment axis: push radially out.
                    // Pick the horizontal direction of least travel.
                    let radial = Vec3::new(local.x, 0.0, local.z);
                    let dir = if radial.length_squared() > EPS_SQ {
                        radial.normalize_or_zero()
                    } else {
                        Vec3::X
                    };
                    let normal = body.transform.rotation * dir;
                    let point = body.transform.transform_point(seg + dir * r);
                    Some((normal, pr + r - diff.length(), point))
                }
            }
            Some(Shape3::ConvexHull { vertices }) | Some(Shape3::ConvexMesh { vertices, .. }) => {
                // Nearest-vertex sphere: conservative, bumpy on coarse
                // hulls. Keeps particles out; exact manifold is future work.
                let mut best: Option<(Vec3, f32, Vec3)> = None;
                for v in vertices.iter() {
                    let w = body.transform.transform_point(*v);
                    let delta = particle - w;
                    let dist = delta.length();
                    if dist < pr {
                        let normal = if dist > EPS { delta / dist } else { Vec3::X };
                        let depth = pr - dist;
                        match best {
                            Some((_, best_depth, _)) if best_depth >= depth => {}
                            _ => best = Some((normal, depth, w)),
                        }
                    }
                }
                best
            }
            None => None,
        }
    }

    /// Advance the fluid one step and couple it to `bodies` (two-way:
    /// particle collisions plus pressure-sampled buoyancy and drag). When
    /// CFL substepping is enabled the pipeline runs `substeps` times with
    /// `dt / substeps`; otherwise it is a single deterministic pass.
    pub(crate) fn step_fluid(
        &mut self,
        bodies: &mut [RigidBody3],
        gravity: Vec3,
        ambient: &crate::ambient::AmbientConfig3,
        time: f32,
        dt: f32,
    ) {
        if !self.config.enabled || self.particles.is_empty() || dt <= 0.0 {
            return;
        }
        let substeps = self.cfl_substeps(dt);
        let sub_dt = dt / substeps as f32;
        for _ in 0..substeps {
            self.substep(bodies, gravity, ambient, time, sub_dt);
        }
    }

    /// Number of CFL substeps (`1` when disabled). See the 2D counterpart.
    fn cfl_substeps(&self, dt: f32) -> u32 {
        let cfg = &self.config;
        let max = cfg.max_substeps.max(1);
        if cfg.cfl <= 0.0 || max <= 1 {
            return 1;
        }
        let h = cfg.smoothing_length.max(EPS);
        let rho0 = cfg.rest_density.max(EPS);
        let c0 = if cfg.gamma == 1.0 {
            cfg.stiffness.max(0.0).sqrt()
        } else {
            (cfg.gamma * cfg.stiffness.max(0.0) * rho0).sqrt()
        };
        let mut vmax = 0.0f32;
        for p in &self.particles {
            vmax = vmax.max(p.velocity.length());
        }
        let speed = c0 + vmax;
        if !speed.is_finite() || speed <= EPS {
            return 1;
        }
        let dt_max = cfg.cfl * h / speed;
        if !dt_max.is_finite() || dt_max <= 0.0 {
            return 1;
        }
        ((dt / dt_max).ceil() as i64).clamp(1, max as i64) as u32
    }

    /// One fluid substep (the full SPH pipeline) over interval `dt`.
    fn substep(
        &mut self,
        bodies: &mut [RigidBody3],
        gravity: Vec3,
        ambient: &crate::ambient::AmbientConfig3,
        time: f32,
        dt: f32,
    ) {
        let cfg = self.config;
        let h = cfg.smoothing_length.max(EPS);
        let m = cfg.particle_mass.max(0.0);

        self.rebuild_grid();

        // 1. Densities (poly6 sum, self included), clamped against
        // collapse singularities (same role as MIN_DENSITY for mass).
        let mut nbrs: Vec<usize> = Vec::new();
        for i in 0..self.particles.len() {
            self.neighbors(i, &mut nbrs);
            let pi = self.particles[i].position;
            let mut rho = 0.0f32;
            for &j in nbrs.iter() {
                let r2 = (self.particles[j].position - pi).length_squared();
                rho += m * Self::poly6(r2, h);
            }
            self.particles[i].density = rho.clamp(0.0, 3.0 * cfg.rest_density.max(EPS));
        }

        // 2. Pressures (linear or Tait equation of state, tension-free).
        for p in self.particles.iter_mut() {
            p.pressure = Self::pressure(p.density, cfg.rest_density, cfg.stiffness, cfg.gamma);
        }

        // 3. Pressure forces (symmetric spiky form) into scratch, plus
        // optional Monaghan artificial viscosity and surface tension.
        let c_s = cfg.stiffness.max(0.0).sqrt();
        self.forces.clear();
        self.forces.resize(self.particles.len(), Vec3::ZERO);
        for i in 0..self.particles.len() {
            self.neighbors(i, &mut nbrs);
            let pi = self.particles[i].position;
            let vi = self.particles[i].velocity;
            let rho_i = self.particles[i].density.max(EPS);
            let p_i = self.particles[i].pressure;
            let mut f = Vec3::ZERO;
            for &j in nbrs.iter() {
                if j == i {
                    continue;
                }
                let pj = self.particles[j];
                let rho_j = pj.density.max(EPS);
                let delta = pi - pj.position;
                let dist = delta.length();
                let grad = Self::spiky_grad(delta, dist, h);
                let mut term = p_i / (rho_i * rho_i) + pj.pressure / (rho_j * rho_j);
                if cfg.artificial_viscosity > 0.0 && dist > EPS {
                    let rho_bar = 0.5 * (rho_i + rho_j);
                    term += Self::artificial_pressure(
                        delta,
                        vi - pj.velocity,
                        dist * dist,
                        h,
                        rho_bar,
                        c_s,
                        cfg.artificial_viscosity,
                    );
                }
                f -= grad * (m * term);
                if cfg.surface_tension > 0.0 && dist > EPS {
                    let coh = Self::cohesion(dist, h);
                    if coh > 0.0 {
                        f -= (delta / dist) * (cfg.surface_tension * m * m * coh);
                    }
                }
            }
            self.forces[i] = f;
        }

        // 4. Integrate particles: gravity + pressure + optional ambient
        // turbulence, XSPH viscosity, speed cap, advect.
        let turb = &ambient.turbulence;
        let turb_on = turb.enabled && turb.amplitude > 0.0;
        for i in 0..self.particles.len() {
            self.neighbors(i, &mut nbrs);
            let rho_i = self.particles[i].density.max(EPS);
            let mut acc = gravity + self.forces[i] / rho_i;
            if turb_on {
                let p = self.particles[i].position * turb.frequency;
                let t = time * turb.time_scale;
                let n = if turb.divergence_free {
                    curl_3(p.x, p.y, p.z, t, turb.seed, turb.octaves)
                } else {
                    Vec3::new(
                        fbm_3(p.x, p.y, p.z, t, turb.seed, turb.octaves),
                        fbm_3(p.x, p.y, p.z, t, turb.seed ^ 0x55aa_55aa, turb.octaves),
                        fbm_3(p.x, p.y, p.z, t, turb.seed ^ 0xaa55_aa55, turb.octaves),
                    )
                };
                acc += n.clamp_length_max(1.0) * turb.amplitude;
            }
            let mut v = self.particles[i].velocity + acc * dt;
            if cfg.viscosity > 0.0 {
                let pi = self.particles[i].position;
                let mut smooth = Vec3::ZERO;
                for &j in nbrs.iter() {
                    if j == i {
                        continue;
                    }
                    let pj = self.particles[j];
                    let rho_bar = 0.5 * (rho_i + pj.density.max(EPS));
                    smooth += (pj.velocity - self.particles[i].velocity)
                        * (m / rho_bar)
                        * Self::poly6((pj.position - pi).length_squared(), h);
                }
                v += smooth * cfg.viscosity;
            }
            let speed = v.length();
            if speed > cfg.max_velocity {
                v *= cfg.max_velocity / speed;
            }
            if !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite() {
                v = Vec3::ZERO;
            }
            self.particles[i].velocity = v;
            let mut pos = self.particles[i].position + v * dt;
            if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
                pos = self.particles[i].position;
            }
            self.particles[i].position = pos;
        }

        // 5. Boundaries: particles vs rigid shapes (two-way impulses).
        // The particle position is re-read for every body: earlier
        // push-outs in this pass must accumulate, never be overwritten
        // from a stale snapshot (corner squeeze-out otherwise).
        let pr = cfg.particle_radius.max(0.0);
        for i in 0..self.particles.len() {
            for body in bodies.iter_mut() {
                let ppos = self.particles[i].position;
                let Some((normal, depth, point)) = self.collide_particle_body(ppos, pr, body)
                else {
                    continue;
                };
                if depth <= 0.0 {
                    continue;
                }
                self.particles[i].position = ppos + normal * depth;
                let vn = self.particles[i].velocity.dot(normal);
                if vn < 0.0 {
                    self.particles[i].velocity -= normal * vn;
                    body.apply_impulse(normal * (vn * m), point);
                    if !body.is_static {
                        body.is_sleeping = false;
                    }
                }
            }
        }
        // 6. Buoyancy + drag over dynamic body surfaces (Archimedes via
        // sampled pressure, Morison-style quadratic drag). Grid rebuilt
        // so sampling sees post-advection positions. Wet bodies stay
        // awake so sub-threshold bobbing never freezes mid-transient.
        self.rebuild_grid();
        for body in bodies.iter_mut() {
            if body.is_static {
                continue;
            }
            self.surface_samples(body);
            if self.sample_scratch.is_empty() {
                continue;
            }
            let drag = self.config.drag.max(0.0);
            let mut wet = false;
            for (point, normal, area) in self.sample_scratch.iter() {
                let p = self.sample_pressure(*point);
                if p <= 0.0 {
                    continue;
                }
                wet = true;
                let r = *point - body.transform.position;
                let v_rel = body.velocity + body.angular_velocity.cross(r) - ambient.drag.wind;
                let speed = v_rel.length();
                let mut force = -*normal * (p * area);
                if speed > EPS {
                    force -= v_rel * (drag * area * speed);
                }
                body.apply_force(force, *point, dt);
            }
            if wet {
                body.is_sleeping = false;
            }
        }
    }
}

impl Default for FluidState3 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Shape3;

    /// Still-water tank: floor + four walls.
    fn tank(world: &mut crate::World3) {
        world
            .add_static_body(
                Vec3::new(2.0, -1.0, 2.0),
                Shape3::Box {
                    half_extents: Vec3::new(3.0, 0.5, 3.0),
                },
            )
            .expect("valid shape");
        for (pos, he) in [
            (Vec3::new(-1.0, 2.0, 2.0), Vec3::new(0.5, 3.5, 3.5)),
            (Vec3::new(5.0, 2.0, 2.0), Vec3::new(0.5, 3.5, 3.5)),
            (Vec3::new(2.0, 2.0, -1.0), Vec3::new(3.5, 3.5, 0.5)),
            (Vec3::new(2.0, 2.0, 5.0), Vec3::new(3.5, 3.5, 0.5)),
        ] {
            world
                .add_static_body(pos, Shape3::Box { half_extents: he })
                .expect("valid shape");
        }
    }

    /// Lattice of particles filling a box region.
    fn lattice(world: &mut crate::World3, origin: Vec3, nx: usize, ny: usize, nz: usize, dx: f32) {
        let mut pts = Vec::new();
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    pts.push(Vec3::new(
                        origin.x + ix as f32 * dx,
                        origin.y + iy as f32 * dx,
                        origin.z + iz as f32 * dx,
                    ));
                }
            }
        }
        let added = world.fluid.add_particles(&pts);
        assert_eq!(added, pts.len());
    }

    fn finite_fluid(world: &crate::World3, ctx: &str) {
        for (i, p) in world.fluid.particles.iter().enumerate() {
            for v in [
                p.position.x,
                p.position.y,
                p.position.z,
                p.velocity.x,
                p.velocity.y,
                p.velocity.z,
                p.density,
            ] {
                assert!(v.is_finite(), "{ctx}: particle {i} non-finite {v}");
            }
        }
        for body in &world.bodies {
            for v in [
                body.transform.position.x,
                body.transform.position.y,
                body.transform.position.z,
                body.velocity.x,
                body.velocity.y,
                body.velocity.z,
            ] {
                assert!(v.is_finite(), "{ctx}: body {} non-finite {v}", body.id);
            }
        }
    }

    #[test]
    fn test_fluid3_kernels_sane() {
        let h = 0.5f32;
        let peak = FluidState3::poly6(0.0, h);
        assert!(peak > 0.0);
        assert_eq!(FluidState3::poly6(h * h, h), 0.0);
        assert_eq!(FluidState3::poly6(h * h * 4.0, h), 0.0);
        assert!(FluidState3::poly6(h * h * 0.25, h) < peak);
        let g = FluidState3::spiky_grad(Vec3::new(0.25, 0.0, 0.0), 0.25, h);
        assert!(g.x < 0.0 && g.y == 0.0 && g.z == 0.0);
        assert_eq!(
            FluidState3::spiky_grad(Vec3::new(h, 0.0, 0.0), h, h),
            Vec3::ZERO
        );
        assert_eq!(FluidState3::spiky_grad(Vec3::ZERO, 0.0, h), Vec3::ZERO);
    }

    #[test]
    fn test_fluid3_dam_break_contained() {
        let mut world = crate::World3::new();
        tank(&mut world);
        // Water column in one corner: 4 x 6 x 4 lattice.
        lattice(&mut world, Vec3::new(0.0, 0.0, 0.0), 4, 6, 4, 0.25);
        let dt = 1.0 / 60.0;
        let mut peak = 0.0f32;
        let mut max_x = 0.0f32;
        for frame in 0..400 {
            world.step(dt);
            for p in world.fluid.particles.iter() {
                peak = peak.max(p.velocity.length());
                max_x = max_x.max(p.position.x);
            }
            if frame % 100 == 0 {
                finite_fluid(&world, "dam break");
            }
        }
        finite_fluid(&world, "dam break final");
        assert!(max_x > 1.5, "dam never collapsed, max_x={max_x}");
        for p in world.fluid.particles.iter() {
            assert!(
                p.position.x > -1.0
                    && p.position.x < 5.0
                    && p.position.z > -1.0
                    && p.position.z < 5.0
                    && p.position.y > -0.6,
                "dam particle escaped: p={}",
                p.position
            );
        }
        assert!(peak < 30.0, "dam break blew up, peak={peak}");
        assert_eq!(world.fluid.particles.len(), 96);
    }

    #[test]
    fn test_fluid3_box_floats() {
        let mut world = crate::World3::new();
        tank(&mut world);
        // Pool walls confine the water to a 2x2 basin: without them the
        // block slumps across the tank floor into a film too shallow to
        // float in (correct dam-break physics, wrong test).
        for (pos, he) in [
            (Vec3::new(0.375, 1.0, 1.5), Vec3::new(0.125, 2.0, 1.25)),
            (Vec3::new(2.625, 1.0, 1.5), Vec3::new(0.125, 2.0, 1.25)),
            (Vec3::new(1.5, 1.0, 0.375), Vec3::new(1.0, 2.0, 0.125)),
            (Vec3::new(1.5, 1.0, 2.625), Vec3::new(1.0, 2.0, 0.125)),
        ] {
            world
                .add_static_body(pos, Shape3::Box { half_extents: he })
                .expect("valid shape");
        }
        // Still water: 6 x 8 x 6 lattice (~1.1 m deep in the pool).
        lattice(&mut world, Vec3::new(0.75, -0.5, 0.75), 6, 8, 6, 0.25);
        let dt = 1.0 / 60.0;
        for _ in 0..200 {
            world.step(dt);
        }
        // Half-density box placed gently just above the surface: it
        // must settle near half-submerged equilibrium instead of
        // punching through to the floor. (Teleporting it submerged
        // would detonate the overlapped particles instead of testing
        // buoyancy.)
        let b = world
            .add_dynamic_body(
                Vec3::new(1.5, 1.4, 1.5),
                Shape3::Box {
                    half_extents: Vec3::new(0.3, 0.3, 0.3),
                },
                0.5,
            )
            .expect("valid shape");
        for frame in 0..400 {
            world.step(dt);
            if frame % 100 == 0 {
                finite_fluid(&world, "float");
            }
        }
        finite_fluid(&world, "float final");
        let y = world.bodies[b].transform.position.y;
        assert!(y > 0.0, "box sank through the water to the floor: y={y}");
        assert!(y < 2.0, "box ejected from the water: y={y}");
    }

    #[test]
    fn test_fluid3_determinism() {
        fn run() -> Vec<Vec3> {
            let mut world = crate::World3::new();
            tank(&mut world);
            lattice(&mut world, Vec3::new(0.0, 0.0, 0.0), 4, 5, 4, 0.25);
            for _ in 0..80 {
                world.step(1.0 / 60.0);
            }
            world.fluid.particles.iter().map(|p| p.position).collect()
        }
        let (a, b) = (run(), run());
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.x.to_bits(), pb.x.to_bits());
            assert_eq!(pa.y.to_bits(), pb.y.to_bits());
            assert_eq!(pa.z.to_bits(), pb.z.to_bits());
        }
    }

    #[test]
    fn test_fluid3_tait_and_cohesion() {
        let (k, rho0) = (100.0f32, 1.0f32);
        assert_eq!(FluidState3::pressure(rho0, rho0, k, 7.0), 0.0);
        assert_eq!(FluidState3::pressure(0.5, rho0, k, 7.0), 0.0);
        let linear = FluidState3::pressure(1.2, rho0, k, 1.0);
        let tait = FluidState3::pressure(1.2, rho0, k, 7.0);
        assert!(tait > linear && linear > 0.0);
        // Cohesion kernel bounded, zero at support.
        let h = 0.5f32;
        assert!(FluidState3::cohesion(0.1, h) > FluidState3::cohesion(0.4, h));
        assert_eq!(FluidState3::cohesion(h, h), 0.0);
        // Artificial viscosity: approaching -> positive, separating -> 0.
        let delta = Vec3::new(1.0, 0.0, 0.0);
        assert!(
            FluidState3::artificial_pressure(
                delta,
                Vec3::new(-2.0, 0.0, 0.0),
                1.0,
                0.5,
                1.0,
                10.0,
                1.0
            ) > 0.0
        );
        assert_eq!(
            FluidState3::artificial_pressure(
                delta,
                Vec3::new(2.0, 0.0, 0.0),
                1.0,
                0.5,
                1.0,
                10.0,
                1.0
            ),
            0.0
        );
    }

    #[test]
    fn test_fluid3_cohesion_attracts() {
        let mut world = crate::World3::new();
        world.gravity = Vec3::ZERO;
        let added = world
            .fluid
            .add_particles(&[Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.4, 0.0, 0.0)]);
        assert_eq!(added, 2);
        world.fluid.config.stiffness = 0.0;
        world.fluid.config.viscosity = 0.0;
        world.fluid.config.surface_tension = 1000.0;
        world.step(1.0 / 60.0);
        assert!(world.fluid.particles[0].velocity.x > 0.0);
        assert!(world.fluid.particles[1].velocity.x < 0.0);
    }

    #[test]
    fn test_fluid3_cfl_substeps() {
        let mut world = crate::World3::new();
        world
            .fluid
            .add_particles(&[Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.25, 0.0, 0.0)]);
        assert_eq!(world.fluid.cfl_substeps(1.0 / 60.0), 1);
        world.fluid.config.gamma = 7.0;
        world.fluid.config.stiffness = 200.0;
        world.fluid.config.cfl = 0.4;
        world.fluid.config.max_substeps = 8;
        world.fluid.particles[0].velocity = Vec3::new(20.0, 0.0, 0.0);
        let n = world.fluid.cfl_substeps(1.0 / 60.0);
        assert!((2..=8).contains(&n), "expected 3D substeps, got {n}");
    }
}
