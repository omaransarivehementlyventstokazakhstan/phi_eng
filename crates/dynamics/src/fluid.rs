//! SPH fluid (2D): weakly-compressible particle fluid coupled to rigid bodies.
//!
//! Particles live in [`FluidState2`], stepped as phase 1b of
//! [`crate::World2::step`] (right after gravity, before broadphase), so
//! the rigid pipeline (broadphase → narrowphase → solve → integrate →
//! sleep) observes post-fluid velocities without any change to its own
//! invariants. All loops run in index order with fixed neighbor-cell
//! order, so fluid stepping is deterministic.
//!
//! Model: density via the 2D poly6 kernel, pressure via a linear
//! equation of state clamped at zero, pressure force via the spiky
//! gradient, and XSPH velocity smoothing for viscosity. Boundaries are
//! the existing rigid bodies: particles collide against circle / box /
//! polygon shapes (reusing the SAT narrowphase), and the reaction
//! impulse is applied to dynamic bodies through
//! [`crate::body::RigidBody2::apply_impulse`] (statics ignore it).
//! Bodies that receive fluid impulses are woken; settled floating
//! bodies may sleep exactly like resting contacts.

use super::body::{RigidBody2, Shape2};
use glam::Vec2;
use std::collections::HashMap;

/// Hard cap on stored particles per 2D fluid state (creation-time guard
/// against runaway allocation; use several states or bigger scenes
/// instead of raising it silently).
pub const MAX_PARTICLES_2D: usize = 4096;

/// Single SPH particle: position, velocity, and per-step density/pressure.
#[derive(Debug, Clone, Copy)]
pub struct Particle2 {
    /// World-space position.
    pub position: Vec2,
    /// World-space velocity.
    pub velocity: Vec2,
    /// Smoothed density from the poly6 sum (recomputed every step).
    pub density: f32,
    /// Pressure from the equation of state (recomputed every step).
    pub pressure: f32,
}

/// Tunables for the 2D SPH fluid. Defaults suit unit-scale scenes
/// (`rest_density = 1`, decimeter particles at 60 Hz).
#[derive(Debug, Clone, Copy)]
pub struct FluidConfig2 {
    /// Uniform particle mass. Should approximate
    /// `rest_density * spacing^2` for a settled lattice.
    pub particle_mass: f32,
    /// Rest density of the fluid.
    pub rest_density: f32,
    /// Kernel support radius. Should cover ~2 particle spacings.
    pub smoothing_length: f32,
    /// Pressure stiffness `k` in `p = max(0, k * (rho - rho0))`.
    /// Must be large enough to support hydrostatic columns: at depth
    /// `d`, equilibrium needs `k * (rho - rho0) = rho0 * g * d`, so
    /// soft `k` caps out (via the density clamp) long before deep
    /// water is supported. Default suits meter-scale 60 Hz scenes.
    pub stiffness: f32,
    /// XSPH smoothing factor in `[0, 1]` (velocity diffusion strength).
    pub viscosity: f32,
    /// Quadratic drag coefficient applied per submerged sample area
    /// (`F = -drag * area * |v| * v`, water-density units). Penalty
    /// boundaries scatter water instead of compressing it, so a fast
    /// body can outrun pressure buildup and punch through its own
    /// cavity; drag caps the entry speed so buoyancy can engage.
    pub drag: f32,
    /// Hard speed cap (safety net against ejection chains).
    pub max_velocity: f32,
    /// Collision radius used against rigid-body shapes.
    pub particle_radius: f32,
    /// Master switch. With no particles the step is a no-op either way.
    pub enabled: bool,
}

impl Default for FluidConfig2 {
    fn default() -> Self {
        Self {
            particle_mass: 0.0625,
            rest_density: 1.0,
            smoothing_length: 0.5,
            stiffness: 100.0,
            viscosity: 0.3,
            drag: 4.0,
            max_velocity: 30.0,
            particle_radius: 0.125,
            enabled: true,
        }
    }
}

/// 2D SPH fluid state owned by [`crate::World2`].
#[derive(Debug, Clone)]
pub struct FluidState2 {
    /// Tunables (see [`FluidConfig2`]).
    pub config: FluidConfig2,
    /// Live particles in insertion order (indices are stable).
    pub particles: Vec<Particle2>,
    /// Uniform-grid neighbor index: cell -> particle indices. Rebuilt
    /// every step; cell vectors are cleared in place (no steady-state
    /// allocation), with vacated cells compacted periodically.
    grid: HashMap<(i32, i32), Vec<usize>>,
    /// Per-particle force scratch, resized as needed.
    forces: Vec<Vec2>,
    /// Reused world-space polygon scratch for boundary tests.
    poly_scratch: Vec<Vec2>,
    /// Reused (point, normal, area) surface samples for buoyancy.
    sample_scratch: Vec<(Vec2, Vec2, f32)>,
    /// Steps since vacated grid cells were compacted.
    clears_since_compact: u32,
}

impl FluidState2 {
    /// Empty fluid with default config.
    pub fn new() -> Self {
        Self {
            config: FluidConfig2::default(),
            particles: Vec::new(),
            grid: HashMap::new(),
            forces: Vec::new(),
            poly_scratch: Vec::new(),
            sample_scratch: Vec::new(),
            clears_since_compact: 0,
        }
    }

    /// Insert particles. Returns the number actually added (capped at
    /// [`MAX_PARTICLES_2D`] total; excess is dropped, never partially
    /// applied per-particle).
    pub fn add_particles(&mut self, positions: &[Vec2]) -> usize {
        let room = MAX_PARTICLES_2D.saturating_sub(self.particles.len());
        let n = room.min(positions.len());
        self.particles.reserve(n);
        for &p in &positions[..n] {
            self.particles.push(Particle2 {
                position: p,
                velocity: Vec2::ZERO,
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

    /// 2D poly6 kernel: `4 / (pi h^8) (h^2 - r^2)^3`, zero beyond `h`.
    fn poly6(r2: f32, h: f32) -> f32 {
        if r2 >= h * h || r2 < 0.0 {
            return 0.0;
        }
        let h2 = h * h;
        let t = h2 - r2;
        use std::f32::consts::PI;
        4.0 / (PI * h2 * h2 * h2 * h2) * t * t * t
    }

    /// 2D spiky gradient: `-30 / (pi h^5) (h - r)^2 * (r_vec / r)`.
    /// Returns zero at and beyond the support radius and at `r == 0`
    /// (no direction), so coincident particles never divide by zero.
    fn spiky_grad(r_vec: Vec2, r: f32, h: f32) -> Vec2 {
        if r >= h || r < 1e-6 {
            return Vec2::ZERO;
        }
        use std::f32::consts::PI;
        let s = -30.0 / (PI * h * h * h * h * h) * (h - r) * (h - r);
        r_vec / r * s
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
            if !p.position.x.is_finite() || !p.position.y.is_finite() {
                continue;
            }
            let key = (
                (p.position.x / h).floor() as i32,
                (p.position.y / h).floor() as i32,
            );
            self.grid.entry(key).or_default().push(i);
        }
    }

    /// Indices of particles that may lie within the support radius of
    /// particle `i`, in deterministic cell order. Includes `i` itself
    /// (the `r == 0` poly6 term is part of the density sum).
    fn neighbors(&self, i: usize, out: &mut Vec<usize>) {
        out.clear();
        let h = self.config.smoothing_length;
        let p = self.particles[i].position;
        let cx = (p.x / h).floor() as i32;
        let cy = (p.y / h).floor() as i32;
        for oy in -1..=1 {
            for ox in -1..=1 {
                if let Some(ids) = self.grid.get(&(cx + ox, cy + oy)) {
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

    /// SPH pressure interpolated at a world-space point:
    /// `p(x) = sum m_j (p_j / rho_j) W(|x - x_j|)`. Zero outside the
    /// fluid (no particles nearby), so bodies above water feel nothing.
    /// Callers must have rebuilt the grid after the latest advection.
    fn sample_pressure(&self, point: Vec2) -> f32 {
        let h = self.config.smoothing_length.max(1e-6);
        let m = self.config.particle_mass.max(0.0);
        let cx = (point.x / h).floor() as i32;
        let cy = (point.y / h).floor() as i32;
        let mut p = 0.0f32;
        for oy in -1..=1 {
            for ox in -1..=1 {
                if let Some(ids) = self.grid.get(&(cx + ox, cy + oy)) {
                    for &j in ids.iter() {
                        let pj = self.particles[j];
                        let r2 = (pj.position - point).length_squared();
                        if r2 < h * h {
                            p += m * (pj.pressure / pj.density.max(1e-6)) * Self::poly6(r2, h);
                        }
                    }
                }
            }
        }
        p.max(0.0)
    }

    /// Surface samples `(point, outward_normal, area)` for buoyancy.
    /// Circles use 12 perimeter samples, boxes 16 (4 per edge).
    /// Polygons are impulse-coupled only (no unambiguous outward
    /// normals without a winding contract), documented at the call site.
    fn surface_samples(&mut self, body: &RigidBody2) {
        self.sample_scratch.clear();
        match &body.shape {
            Some(Shape2::Circle { radius }) => {
                let n = 12;
                let area = 2.0 * std::f32::consts::PI * radius / n as f32;
                for k in 0..n {
                    let a = 2.0 * std::f32::consts::PI * k as f32 / n as f32;
                    let local = Vec2::new(a.cos(), a.sin());
                    self.sample_scratch.push((
                        body.transform.transform_point(local * radius),
                        body.transform.rotation * local,
                        area,
                    ));
                }
            }
            Some(Shape2::Box { half_extents }) => {
                let he = *half_extents;
                let perimeter = 4.0 * (he.x + he.y);
                let area = perimeter / 16.0;
                let edges = [
                    (Vec2::new(0.0, -1.0), Vec2::new(-he.x, -he.y), Vec2::X),
                    (Vec2::new(0.0, 1.0), Vec2::new(-he.x, he.y), Vec2::X),
                    (Vec2::new(-1.0, 0.0), Vec2::new(-he.x, -he.y), Vec2::Y),
                    (Vec2::new(1.0, 0.0), Vec2::new(he.x, -he.y), Vec2::Y),
                ];
                for (normal, origin, tangent) in edges {
                    for k in 0..4 {
                        let t = (k as f32 + 0.5) / 4.0;
                        let extent = if tangent == Vec2::X { he.x } else { he.y };
                        let local = origin + tangent * (2.0 * extent * t);
                        self.sample_scratch.push((
                            body.transform.transform_point(local),
                            body.transform.rotation * normal,
                            area,
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    /// Particle-vs-body boundary test. Returns the outward normal
    /// (body -> particle), penetration depth, and a contact point on
    /// the body surface. `None` when separated.
    fn collide_particle_body(
        &mut self,
        particle: Vec2,
        pr: f32,
        body: &RigidBody2,
    ) -> Option<(Vec2, f32, Vec2)> {
        match &body.shape {
            Some(Shape2::Circle { radius }) => {
                let delta = particle - body.transform.position;
                let dist = delta.length();
                if dist >= radius + pr {
                    return None;
                }
                let normal = if dist > 1e-6 { delta / dist } else { Vec2::X };
                let depth = radius + pr - dist;
                let point = body.transform.position + normal * radius;
                Some((normal, depth, point))
            }
            Some(Shape2::Box { half_extents }) => {
                let he = *half_extents;
                let rel = particle - body.transform.position;
                // To local frame (rotation is orthonormal: transpose == inverse).
                let local = Vec2::new(
                    body.transform.rotation.x_axis.x * rel.x
                        + body.transform.rotation.y_axis.x * rel.y,
                    body.transform.rotation.x_axis.y * rel.x
                        + body.transform.rotation.y_axis.y * rel.y,
                );
                let clamped = Vec2::new(local.x.clamp(-he.x, he.x), local.y.clamp(-he.y, he.y));
                let diff = local - clamped;
                if diff.length_squared() > 1e-12 {
                    // Center outside: closest surface point.
                    let dist = diff.length();
                    if dist >= pr {
                        return None;
                    }
                    let local_n = diff / dist;
                    let normal = body.transform.rotation * local_n;
                    let depth = pr - dist;
                    let point = body.transform.transform_point(clamped);
                    Some((normal, depth, point))
                } else {
                    // Center inside: eject along least-penetration axis.
                    let dx = he.x - local.x.abs();
                    let dy = he.y - local.y.abs();
                    let (local_n, depth) = if dx < dy {
                        (Vec2::new(local.x.signum().copysign(1.0), 0.0), pr + dx)
                    } else {
                        (Vec2::new(0.0, local.y.signum().copysign(1.0)), pr + dy)
                    };
                    let normal = body.transform.rotation * local_n;
                    let point = body.transform.transform_point(clamped);
                    Some((normal, depth, point))
                }
            }
            Some(Shape2::Polygon { vertices }) => {
                self.poly_scratch.clear();
                self.poly_scratch.reserve(vertices.len());
                for v in vertices {
                    self.poly_scratch.push(body.transform.transform_point(*v));
                }
                let (axis, penetration) =
                    phieng_collision::sat::sat_circle_polygon(particle, pr, &self.poly_scratch)?;
                let centroid = phieng_collision::sat::polygon_centroid(&self.poly_scratch);
                let normal = phieng_collision::sat::orient_normal(axis, centroid, particle);
                let point = particle - normal * pr;
                Some((normal, penetration, point))
            }
            None => None,
        }
    }

    /// Advance the fluid one step and couple it to `bodies` (two-way:
    /// particles collide with shapes; reactions wake and push dynamics).
    /// Deterministic: particles and bodies are visited in index order
    /// with fixed neighbor-cell order.
    pub(crate) fn step_fluid(&mut self, bodies: &mut [RigidBody2], gravity: Vec2, dt: f32) {
        if !self.config.enabled || self.particles.is_empty() || dt <= 0.0 {
            return;
        }
        let cfg = self.config;
        let h = cfg.smoothing_length.max(1e-6);
        let m = cfg.particle_mass.max(0.0);

        self.rebuild_grid();

        // 1. Densities (poly6 sum, self included).
        let mut nbrs: Vec<usize> = Vec::new();
        for i in 0..self.particles.len() {
            self.neighbors(i, &mut nbrs);
            let pi = self.particles[i].position;
            let mut rho = 0.0f32;
            for &j in nbrs.iter() {
                let r2 = (self.particles[j].position - pi).length_squared();
                rho += m * Self::poly6(r2, h);
            }
            // Safety net against collapse singularities (same role as
            // MIN_DENSITY for rigid mass): densities above 3x rest are
            // unphysical packing, not information.
            self.particles[i].density = rho.clamp(0.0, 3.0 * cfg.rest_density.max(1e-6));
        }

        // 2. Pressures (linear equation of state, tension-free).
        for p in self.particles.iter_mut() {
            p.pressure = (cfg.stiffness * (p.density - cfg.rest_density)).max(0.0);
        }

        // 3. Pressure forces (symmetric spiky form) into scratch.
        self.forces.clear();
        self.forces.resize(self.particles.len(), Vec2::ZERO);
        for i in 0..self.particles.len() {
            self.neighbors(i, &mut nbrs);
            let pi = self.particles[i].position;
            let rho_i = self.particles[i].density.max(1e-6);
            let p_i = self.particles[i].pressure;
            let mut f = Vec2::ZERO;
            for &j in nbrs.iter() {
                if j == i {
                    continue;
                }
                let pj = self.particles[j];
                let rho_j = pj.density.max(1e-6);
                let delta = pi - pj.position;
                let dist = delta.length();
                let grad = Self::spiky_grad(delta, dist, h);
                f -= grad * (m * (p_i / (rho_i * rho_i) + pj.pressure / (rho_j * rho_j)));
            }
            self.forces[i] = f;
        }

        // 4. Integrate particles: gravity + pressure, XSPH viscosity,
        // speed cap, advect.
        for i in 0..self.particles.len() {
            self.neighbors(i, &mut nbrs);
            let rho_i = self.particles[i].density.max(1e-6);
            let mut v = self.particles[i].velocity + (gravity + self.forces[i] / rho_i) * dt;
            // XSPH velocity smoothing against pre-update neighbors.
            if cfg.viscosity > 0.0 {
                let pi = self.particles[i].position;
                let mut smooth = Vec2::ZERO;
                for &j in nbrs.iter() {
                    if j == i {
                        continue;
                    }
                    let pj = self.particles[j];
                    let rho_bar = 0.5 * (rho_i + pj.density.max(1e-6));
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
            if !v.x.is_finite() || !v.y.is_finite() {
                v = Vec2::ZERO;
            }
            self.particles[i].velocity = v;
            let mut pos = self.particles[i].position + v * dt;
            if !pos.x.is_finite() || !pos.y.is_finite() {
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
                // Push the particle out of the shape.
                self.particles[i].position = ppos + normal * depth;
                // Kill approach velocity (inelastic boundary).
                let vn = self.particles[i].velocity.dot(normal);
                if vn < 0.0 {
                    self.particles[i].velocity -= normal * vn;
                    // Reaction on the body (statics ignore it internally).
                    body.apply_impulse(normal * (vn * m), point);
                    if !body.is_static {
                        body.is_sleeping = false;
                    }
                }
            }
        }
        // 6. Buoyancy + drag: sample the SPH pressure field over
        // dynamic body surfaces and apply hydrostatic force (Archimedes,
        // integrated per sample) plus Morison-style quadratic drag on
        // the relative surface velocity. The grid is rebuilt first so
        // sampling sees the post-advection positions. Bodies with any
        // wet sample stay awake: sub-threshold bobbing must not freeze
        // mid-transient.
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
                // Relative surface velocity (translation + rotation).
                let r = *point - body.transform.position;
                let v_rel = body.velocity
                    + Vec2::new(-body.angular_velocity * r.y, body.angular_velocity * r.x);
                let speed = v_rel.length();
                let mut force = -*normal * (p * area);
                if speed > 1e-6 {
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

impl Default for FluidState2 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Shape2;

    /// Still-water tank: floor + side walls.
    fn tank(world: &mut crate::World2) {
        world
            .add_static_body(
                Vec2::new(2.0, -1.0),
                Shape2::Box {
                    half_extents: Vec2::new(3.0, 0.5),
                },
            )
            .expect("valid shape");
        for x in [-1.0f32, 5.0] {
            world
                .add_static_body(
                    Vec2::new(x, 2.0),
                    Shape2::Box {
                        half_extents: Vec2::new(0.5, 3.5),
                    },
                )
                .expect("valid shape");
        }
    }

    /// Lattice of particles filling a rectangle.
    fn lattice(world: &mut crate::World2, x0: f32, y0: f32, nx: usize, ny: usize, dx: f32) {
        let mut pts = Vec::new();
        for iy in 0..ny {
            for ix in 0..nx {
                pts.push(Vec2::new(x0 + ix as f32 * dx, y0 + iy as f32 * dx));
            }
        }
        let added = world.fluid.add_particles(&pts);
        assert_eq!(added, pts.len());
    }

    fn finite_fluid(world: &crate::World2, ctx: &str) {
        for (i, p) in world.fluid.particles.iter().enumerate() {
            for v in [
                p.position.x,
                p.position.y,
                p.velocity.x,
                p.velocity.y,
                p.density,
            ] {
                assert!(v.is_finite(), "{ctx}: particle {i} non-finite {v}");
            }
        }
        for body in &world.bodies {
            for v in [
                body.transform.position.x,
                body.transform.position.y,
                body.velocity.x,
                body.velocity.y,
            ] {
                assert!(v.is_finite(), "{ctx}: body {} non-finite {v}", body.id);
            }
        }
    }

    #[test]
    fn test_fluid_kernels_sane() {
        let h = 0.5f32;
        // poly6 peaks at r=0 and vanishes at support.
        let peak = FluidState2::poly6(0.0, h);
        assert!(peak > 0.0);
        assert_eq!(FluidState2::poly6(h * h, h), 0.0);
        assert_eq!(FluidState2::poly6(h * h * 4.0, h), 0.0);
        assert!(FluidState2::poly6(h * h * 0.25, h) < peak);
        // Spiky gradient points inward along r and vanishes at support/center.
        let g = FluidState2::spiky_grad(Vec2::new(0.25, 0.0), 0.25, h);
        assert!(g.x < 0.0 && g.y == 0.0);
        assert_eq!(FluidState2::spiky_grad(Vec2::new(h, 0.0), h, h), Vec2::ZERO);
        assert_eq!(FluidState2::spiky_grad(Vec2::ZERO, 0.0, h), Vec2::ZERO);
    }

    #[test]
    fn test_fluid_dam_break_contained() {
        let mut world = crate::World2::new();
        tank(&mut world);
        // Water column on the left: 8 x 12 lattice.
        lattice(&mut world, 0.0, 0.0, 8, 12, 0.25);
        let dt = 1.0 / 60.0;
        let mut peak = 0.0f32;
        let mut max_x = 0.0f32;
        for frame in 0..600 {
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
        // The column must collapse rightward, stay in the tank, stay calm.
        assert!(max_x > 2.5, "dam never collapsed, max_x={max_x}");
        for p in world.fluid.particles.iter() {
            assert!(
                p.position.x > -1.0 && p.position.x < 5.0 && p.position.y > -0.6,
                "dam particle escaped: p={}",
                p.position
            );
        }
        assert!(peak < 30.0, "dam break blew up, peak={peak}");
        assert_eq!(world.fluid.particles.len(), 96);
    }

    #[test]
    fn test_fluid_box_floats() {
        let mut world = crate::World2::new();
        tank(&mut world);
        // Still water: 12 x 6 lattice (surface near y = 1.0).
        lattice(&mut world, -0.5, -0.5, 12, 6, 0.25);
        let dt = 1.0 / 60.0;
        for _ in 0..300 {
            world.step(dt);
        }
        // Half-density box placed gently just above the surface: it must
        // settle near half-submerged (center ~ surface), not punch
        // through to the floor and not eject.
        let b = world
            .add_dynamic_body(
                Vec2::new(2.0, 1.6),
                Shape2::Box {
                    half_extents: Vec2::new(0.3, 0.3),
                },
                0.5,
            )
            .expect("valid shape");
        for frame in 0..600 {
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
    fn test_fluid_determinism() {
        fn run() -> Vec<Vec2> {
            let mut world = crate::World2::new();
            tank(&mut world);
            lattice(&mut world, 0.0, 0.0, 6, 8, 0.25);
            for _ in 0..120 {
                world.step(1.0 / 60.0);
            }
            world.fluid.particles.iter().map(|p| p.position).collect()
        }
        let (a, b) = (run(), run());
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.x.to_bits(), pb.x.to_bits());
            assert_eq!(pa.y.to_bits(), pb.y.to_bits());
        }
    }
}
