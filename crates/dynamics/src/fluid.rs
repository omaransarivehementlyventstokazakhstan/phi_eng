//! SPH fluid (2D): weakly-compressible particle fluid coupled to rigid bodies.
//!
//! Particles live in [`FluidState2`], stepped as phase 1b of
//! [`crate::World2::step`] (right after gravity, before broadphase), so
//! the rigid pipeline (broadphase â†’ narrowphase â†’ solve â†’ integrate â†’
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
use crate::fxhash::FxHashMap;
use crate::noise::{curl_2, fbm_2};
use glam::Vec2;
use phieng_math::{EPS, EPS_SQ};
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
    /// Per-particle mass. Uniform `config.particle_mass` unless adaptive
    /// resolution split/merged the particle (children carry a share).
    pub mass: f32,
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
    /// Pressure stiffness `k` in `p = max(0, k * (rho - rho0))` (linear
    /// EOS, `gamma == 1`). Must be large enough to support hydrostatic
    /// columns: at depth `d`, equilibrium needs `k * (rho - rho0) =
    /// rho0 * g * d`. In Tait mode (`gamma > 1`) this acts as the
    /// `B`-scale of `p = max(0, k*rho0*((rho/rho0)^gamma - 1))`.
    pub stiffness: f32,
    /// Equation-of-state exponent. `1.0` selects the legacy linear EOS
    /// (bit-identical); values `> 1` select the Tait/WCSPH form
    /// `p = max(0, k*rho0*((rho/rho0)^gamma - 1))`, which supports deep
    /// columns with far less artificial stiffness. Classic water value
    /// is `7.0`.
    pub gamma: f32,
    /// Monaghan artificial-viscosity `alpha` in `[0, 2]` (default `0` =
    /// off). Adds a pairwise viscous pressure `Π = -alpha*c*mu/rho_bar`
    /// that damps approaching particles; `c` is approximated by
    /// `sqrt(stiffness)`. Stable at low XSPH settings.
    pub artificial_viscosity: f32,
    /// Akinci-style surface-tension (cohesion) strength `kappa` (default
    /// `0` = off). Neighbors feel a bounded attractive force that holds
    /// droplets together and rounds free surfaces; balanced by pressure.
    pub surface_tension: f32,
    /// Courant–Friedrichs–Lewy factor for adaptive substepping (default
    /// `0.0` = off, a single pass). Set to ~`0.4` to subdivide when the
    /// signal speed would otherwise violate `dt < cfl*h/(c0+v)` — the way
    /// to run stiff Tait configs stably at 60 Hz.
    pub cfl: f32,
    /// Maximum substeps per world step when `cfl` is enabled (default
    /// `1` = off, i.e. a single pass). Clamps runaway cost.
    pub max_substeps: u32,
    /// Adaptive resolution (default off): under-dense particles split and
    /// over-dense neighbors merge, conserving total mass. Changes the
    /// particle count dynamically (capped at [`MAX_PARTICLES_2D`]).
    pub adaptive_resolution: bool,
    /// Split a particle when `density < split_density_ratio * rest_density`
    /// (isolated / free-surface particle). Finite, `> 0`.
    pub split_density_ratio: f32,
    /// Merge a particle into a close neighbor when both exceed
    /// `merge_density_ratio * rest_density`. Finite, `> 1`.
    pub merge_density_ratio: f32,
    /// Boundary particles (default off): static bodies are sampled into
    /// fixed ghost particles that exert SPH pressure on the fluid, giving
    /// pressure-based walls that reduce tunneling vs the penalty boundary
    /// alone. Regenerated every substep; never persisted (statics don't
    /// move, so the set is deterministic from the current bodies).
    pub boundary_particles: bool,
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
            gamma: 1.0,
            artificial_viscosity: 0.0,
            surface_tension: 0.0,
            cfl: 0.0,
            max_substeps: 1,
            adaptive_resolution: false,
            split_density_ratio: 0.5,
            merge_density_ratio: 1.6,
            boundary_particles: false,
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
    /// Boundary (ghost) particles for pressure coupling. Regenerated each
    /// substep from the current static bodies; never persisted.
    boundary: Vec<Particle2>,
    /// Cell -> boundary indices (rebuilt with `boundary`).
    boundary_grid: FxHashMap<(i32, i32), Vec<usize>>,
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
            boundary: Vec::new(),
            boundary_grid: FxHashMap::default(),
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
                mass: self.config.particle_mass.max(0.0),
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
        if r >= h || r < EPS {
            return Vec2::ZERO;
        }
        use std::f32::consts::PI;
        let s = -30.0 / (PI * h * h * h * h * h) * (h - r) * (h - r);
        r_vec / r * s
    }

    /// Pressure from the configured equation of state (tension-free).
    /// `gamma == 1` is the legacy linear form (bit-identical); `gamma > 1`
    /// is the Tait/WCSPH form `B*((rho/rho0)^gamma - 1)` with `B = k*rho0`.
    fn pressure(density: f32, rest_density: f32, stiffness: f32, gamma: f32) -> f32 {
        if gamma == 1.0 {
            (stiffness * (density - rest_density)).max(0.0)
        } else {
            let rho0 = rest_density.max(EPS);
            (stiffness * rho0 * ((density / rho0).powf(gamma) - 1.0)).max(0.0)
        }
    }

    /// Monaghan artificial-viscosity pressure term for an approaching
    /// pair (`(v_i - v_j) · (x_i - x_j) < 0`), else zero. `r_sq` is
    /// `|x_i - x_j|^2`; `c` is the (approximate) speed of sound.
    fn artificial_pressure(
        delta: Vec2,
        v_rel: Vec2,
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

    /// Bounded cohesion kernel (simplified Akinci): `1 - r/h` on `[0, h)`.
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
        let h = self.config.smoothing_length.max(EPS);
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
                            p += pj.mass * (pj.pressure / pj.density.max(EPS)) * Self::poly6(r2, h);
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
                let normal = if dist > EPS { delta / dist } else { Vec2::X };
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
                if diff.length_squared() > EPS_SQ {
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
    /// When CFL substepping is enabled (`cfl > 0` and `max_substeps > 1`)
    /// the pipeline runs `substeps` times with `dt / substeps`, so stiff
    /// (high sound-speed) Tait configs stay stable; otherwise it is a
    /// single deterministic pass (bit-identical to legacy behavior).
    pub(crate) fn step_fluid(
        &mut self,
        bodies: &mut [RigidBody2],
        gravity: Vec2,
        ambient: &crate::ambient::AmbientConfig2,
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

    /// Number of CFL substeps for this step (`1` when disabled). Uses the
    /// sound speed `c0 = sqrt(gamma*k*rho0)` (or `sqrt(k)` for the linear
    /// EOS) plus the peak particle speed against `dt < cfl*h/(c0+v)`.
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

    /// Recompute per-particle density (poly6 sum, self included, clamped
    /// against collapse) and pressure (configured EOS, tension-free).
    fn compute_density_pressure(&mut self, h: f32, nbrs: &mut Vec<usize>) {
        let cfg = self.config;
        for i in 0..self.particles.len() {
            self.neighbors(i, nbrs);
            let pi = self.particles[i].position;
            let mut rho = 0.0f32;
            for &j in nbrs.iter() {
                let r2 = (self.particles[j].position - pi).length_squared();
                rho += self.particles[j].mass * Self::poly6(r2, h);
            }
            // Safety net against collapse singularities (same role as
            // MIN_DENSITY for rigid mass): densities above 3x rest are
            // unphysical packing, not information.
            self.particles[i].density = rho.clamp(0.0, 3.0 * cfg.rest_density.max(EPS));
        }
        for p in self.particles.iter_mut() {
            p.pressure = Self::pressure(p.density, cfg.rest_density, cfg.stiffness, cfg.gamma);
        }
    }

    /// Regenerate boundary (ghost) particles from the current static bodies.
    /// Deterministic surface sampling at `spacing`; each ghost carries the
    /// fluid's rest density and unit mass so it contributes to the fluid
    /// density and exerts SPH pressure. Statics never move, so the set is
    /// fully determined by the body list.
    fn refresh_boundaries(&mut self, bodies: &[RigidBody2], spacing: f32) {
        self.boundary.clear();
        let rest = self.config.rest_density;
        let mass = self.config.particle_mass.max(EPS);
        let spacing = spacing.max(EPS);
        for body in bodies.iter().filter(|b| b.is_static) {
            match &body.shape {
                Some(Shape2::Circle { radius }) => {
                    let n = ((2.0 * std::f32::consts::PI * radius) / spacing).ceil() as usize;
                    let n = n.max(1);
                    for k in 0..n {
                        let a = 2.0 * std::f32::consts::PI * k as f32 / n as f32;
                        let local = Vec2::new(a.cos(), a.sin()) * radius;
                        self.boundary.push(Particle2 {
                            position: body.transform.transform_point(local),
                            velocity: Vec2::ZERO,
                            density: rest,
                            pressure: 0.0,
                            mass,
                        });
                    }
                }
                Some(Shape2::Box { half_extents }) => {
                    let he = *half_extents;
                    let edges = [
                        (Vec2::new(0.0, -1.0), Vec2::new(-he.x, -he.y), Vec2::Y),
                        (Vec2::new(0.0, 1.0), Vec2::new(-he.x, he.y), Vec2::Y),
                        (Vec2::new(-1.0, 0.0), Vec2::new(-he.x, -he.y), Vec2::X),
                        (Vec2::new(1.0, 0.0), Vec2::new(he.x, -he.y), Vec2::X),
                    ];
                    for (_normal, origin, tangent) in edges {
                        let extent = if tangent == Vec2::X { he.x } else { he.y };
                        let n = ((2.0 * extent) / spacing).ceil() as usize;
                        let n = n.max(1);
                        for k in 0..n {
                            let t = (k as f32 + 0.5) / n as f32;
                            let local = origin + tangent * (2.0 * extent * t);
                            self.boundary.push(Particle2 {
                                position: body.transform.transform_point(local),
                                velocity: Vec2::ZERO,
                                density: rest,
                                pressure: 0.0,
                                mass,
                            });
                        }
                    }
                }
                Some(Shape2::Polygon { vertices }) => {
                    for e in 0..vertices.len() {
                        let a = vertices[e];
                        let b = vertices[(e + 1) % vertices.len()];
                        let edge = b - a;
                        let len = edge.length();
                        if len < EPS {
                            continue;
                        }
                        let n = (len / spacing).ceil() as usize;
                        let n = n.max(1);
                        for k in 0..n {
                            let t = (k as f32 + 0.5) / n as f32;
                            let local = a + edge * t;
                            self.boundary.push(Particle2 {
                                position: body.transform.transform_point(local),
                                velocity: Vec2::ZERO,
                                density: rest,
                                pressure: 0.0,
                                mass,
                            });
                        }
                    }
                }
                None => {}
            }
        }
    }

    /// Rebuild the boundary neighbor grid.
    fn rebuild_boundary_grid(&mut self, h: f32) {
        self.boundary_grid.clear();
        for (i, p) in self.boundary.iter().enumerate() {
            if !p.position.x.is_finite() || !p.position.y.is_finite() {
                continue;
            }
            let key = (
                (p.position.x / h).floor() as i32,
                (p.position.y / h).floor() as i32,
            );
            self.boundary_grid.entry(key).or_default().push(i);
        }
    }

    /// Boundary indices within the support radius of a world-space point
    /// (used to apply ghost pressure to a fluid particle).
    fn boundary_near(&self, point: Vec2, out: &mut Vec<usize>) {
        out.clear();
        let h = self.config.smoothing_length;
        let cx = (point.x / h).floor() as i32;
        let cy = (point.y / h).floor() as i32;
        for oy in -1..=1 {
            for ox in -1..=1 {
                if let Some(ids) = self.boundary_grid.get(&(cx + ox, cy + oy)) {
                    for &j in ids.iter() {
                        let d2 = (self.boundary[j].position - point).length_squared();
                        if d2 < h * h {
                            out.push(j);
                        }
                    }
                }
            }
        }
    }

    /// Fluid-particle indices within the support radius of a world-space
    /// point (used to compute a boundary particle's density from fluid).
    fn fluid_near(&self, point: Vec2, out: &mut Vec<usize>) {
        out.clear();
        let h = self.config.smoothing_length;
        let cx = (point.x / h).floor() as i32;
        let cy = (point.y / h).floor() as i32;
        for oy in -1..=1 {
            for ox in -1..=1 {
                if let Some(ids) = self.grid.get(&(cx + ox, cy + oy)) {
                    for &j in ids.iter() {
                        let d2 = (self.particles[j].position - point).length_squared();
                        if d2 < h * h {
                            out.push(j);
                        }
                    }
                }
            }
        }
    }

    /// Boundary density/pressure from nearby fluid particles (two-pass
    /// WCSPH): a ghost repels only when fluid is packed against it.
    fn compute_boundary_density_pressure(&mut self, h: f32, nbrs: &mut Vec<usize>) {
        let cfg = self.config;
        for bi in 0..self.boundary.len() {
            let bp = self.boundary[bi].position;
            self.fluid_near(bp, nbrs);
            let mut rho = 0.0f32;
            for &j in nbrs.iter() {
                let r2 = (self.particles[j].position - bp).length_squared();
                rho += self.particles[j].mass * Self::poly6(r2, h);
            }
            self.boundary[bi].density = rho.clamp(0.0, 3.0 * cfg.rest_density.max(EPS));
            self.boundary[bi].pressure =
                Self::pressure(rho, cfg.rest_density, cfg.stiffness, cfg.gamma);
        }
    }

    /// Adaptive-resolution pass (opt-in via `adaptive_resolution`): merge
    /// over-dense close pairs (mass/momentum conserved), then split
    /// under-dense particles into two half-mass children. Bounded by the
    /// particle cap and a minimum child mass (so refinement terminates).
    /// Deterministic (index order, fixed offsets). O(n²) merge scan —
    /// intended for modest particle counts.
    fn resample(&mut self, h: f32) {
        let cfg = self.config;
        let rest = cfg.rest_density.max(EPS);
        let split_lo = cfg.split_density_ratio * rest;
        let merge_hi = cfg.merge_density_ratio * rest;
        let min_mass = cfg.particle_mass.max(EPS) * 0.125;

        // Merge: combine a high-density particle with a close low-index
        // high-density neighbor.
        let n = self.particles.len();
        let mut absorbed = vec![false; n];
        let merge_dist_sq = (0.25 * h) * (0.25 * h);
        for i in 0..n {
            if absorbed[i] || self.particles[i].density <= merge_hi {
                continue;
            }
            for j in (0..i).rev() {
                if absorbed[j] || self.particles[j].density <= merge_hi {
                    continue;
                }
                let d2 = (self.particles[i].position - self.particles[j].position).length_squared();
                if d2 < merge_dist_sq {
                    let mi = self.particles[i].mass;
                    let mj = self.particles[j].mass;
                    let mt = mi + mj;
                    if mt > EPS {
                        self.particles[i].velocity = (self.particles[i].velocity * mi
                            + self.particles[j].velocity * mj)
                            / mt;
                        self.particles[i].mass = mt;
                    }
                    absorbed[j] = true;
                    break;
                }
            }
        }
        if absorbed.iter().any(|&a| a) {
            let mut kept = Vec::with_capacity(n);
            for (idx, p) in std::mem::take(&mut self.particles).into_iter().enumerate() {
                if !absorbed[idx] {
                    kept.push(p);
                }
            }
            self.particles = kept;
        }

        // Split: under-dense particles with room to refine spawn a child.
        let mut out: Vec<Particle2> = Vec::with_capacity(self.particles.len());
        for p in std::mem::take(&mut self.particles) {
            if p.density < split_lo && p.mass > min_mass && out.len() + 1 < MAX_PARTICLES_2D {
                let half = p.mass * 0.5;
                let off = Vec2::new(0.25 * h, 0.0);
                out.push(Particle2 {
                    mass: half,
                    position: p.position + off,
                    ..p
                });
                out.push(Particle2 {
                    mass: half,
                    position: p.position - off,
                    ..p
                });
            } else {
                out.push(p);
            }
        }
        self.particles = out;
    }

    /// One fluid substep (the full SPH pipeline) over interval `dt`.
    /// Deterministic: particles and bodies are visited in index order with
    /// fixed neighbor-cell order.
    fn substep(
        &mut self,
        bodies: &mut [RigidBody2],
        gravity: Vec2,
        ambient: &crate::ambient::AmbientConfig2,
        time: f32,
        dt: f32,
    ) {
        let cfg = self.config;
        let h = cfg.smoothing_length.max(EPS);

        self.rebuild_grid();

        // 1+2. Densities (poly6 sum) and pressures (EOS, tension-free).
        let mut nbrs: Vec<usize> = Vec::new();
        let mut bnbrs: Vec<usize> = Vec::new();
        self.compute_density_pressure(h, &mut nbrs);

        // 2b. Adaptive resolution (opt-in): resample, rebuild the grid,
        // then recompute densities/pressures for the new particle set so
        // force/integrate see fresh values.
        if cfg.adaptive_resolution {
            self.resample(h);
            self.rebuild_grid();
            self.compute_density_pressure(h, &mut nbrs);
        }

        // 2c. Boundary particles (opt-in): regenerate ghosts from static
        // bodies, build their grid, and compute their density/pressure from
        // the fluid (two-pass WCSPH).
        if cfg.boundary_particles {
            self.refresh_boundaries(bodies, h * 0.5);
            self.rebuild_boundary_grid(h);
            self.compute_boundary_density_pressure(h, &mut nbrs);
        }

        // 3. Pressure forces (symmetric spiky form) into scratch, plus
        // optional Monaghan artificial viscosity and surface tension.
        let c_s = cfg.stiffness.max(0.0).sqrt();
        self.forces.clear();
        self.forces.resize(self.particles.len(), Vec2::ZERO);
        for i in 0..self.particles.len() {
            self.neighbors(i, &mut nbrs);
            let pi = self.particles[i].position;
            let vi = self.particles[i].velocity;
            let rho_i = self.particles[i].density.max(EPS);
            let p_i = self.particles[i].pressure;
            let mut f = Vec2::ZERO;
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
                f -= grad * (pj.mass * term);
                if cfg.surface_tension > 0.0 && dist > EPS {
                    let coh = Self::cohesion(dist, h);
                    if coh > 0.0 {
                        f -= (delta / dist)
                            * (cfg.surface_tension * self.particles[i].mass * pj.mass * coh);
                    }
                }
            }
            // Boundary pressure force (ghost particles): a ghost repels only
            // when fluid is packed against it (its pressure is fluid-derived).
            if cfg.boundary_particles {
                self.boundary_near(pi, &mut bnbrs);
                for &bk in bnbrs.iter() {
                    let bp = self.boundary[bk];
                    let delta = pi - bp.position;
                    let dist = delta.length();
                    let grad = Self::spiky_grad(delta, dist, h);
                    let rho_b = bp.density.max(EPS);
                    f -= grad * (bp.mass * (p_i / (rho_i * rho_i) + bp.pressure / (rho_b * rho_b)));
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
                    curl_2(p.x, p.y, t, turb.seed, turb.octaves)
                } else {
                    Vec2::new(
                        fbm_2(p.x, p.y, t, turb.seed, turb.octaves),
                        fbm_2(p.x, p.y, t, turb.seed ^ 0x55aa_55aa, turb.octaves),
                    )
                };
                acc += n.clamp_length_max(1.0) * turb.amplitude;
            }
            let mut v = self.particles[i].velocity + acc * dt;
            // XSPH velocity smoothing against pre-update neighbors.
            if cfg.viscosity > 0.0 {
                let pi = self.particles[i].position;
                let mut smooth = Vec2::ZERO;
                for &j in nbrs.iter() {
                    if j == i {
                        continue;
                    }
                    let pj = self.particles[j];
                    let rho_bar = 0.5 * (rho_i + pj.density.max(EPS));
                    smooth += (pj.velocity - self.particles[i].velocity)
                        * (pj.mass / rho_bar)
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
                    body.apply_impulse(normal * (vn * self.particles[i].mass), point);
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
                // Relative surface velocity (translation + rotation) against
                // the ambient wind, so a current sweeps floating bodies.
                let r = *point - body.transform.position;
                let v_rel = body.velocity
                    + Vec2::new(-body.angular_velocity * r.y, body.angular_velocity * r.x)
                    - ambient.drag.wind;
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

    #[test]
    fn test_fluid_tait_eos_pressure() {
        let (k, rho0) = (100.0f32, 1.0f32);
        // At and below rest density: tension-free (zero).
        assert_eq!(FluidState2::pressure(rho0, rho0, k, 7.0), 0.0);
        assert_eq!(FluidState2::pressure(0.5, rho0, k, 7.0), 0.0);
        // Above rest density: Tait pressure exceeds the linear EOS and
        // grows faster (gamma > 1).
        let linear = FluidState2::pressure(1.2, rho0, k, 1.0);
        let tait = FluidState2::pressure(1.2, rho0, k, 7.0);
        assert!(linear > 0.0);
        assert!(
            tait > linear,
            "Tait {tait} should exceed linear {linear} at rho=1.2"
        );
        // gamma == 1 is exactly the legacy linear form.
        assert_eq!(
            FluidState2::pressure(1.2, rho0, k, 1.0).to_bits(),
            (k * (1.2 - rho0)).to_bits()
        );
    }

    #[test]
    fn test_fluid_artificial_pressure_sign() {
        let delta = Vec2::new(1.0, 0.0);
        // Approaching (v_rel · delta < 0) -> positive viscous pressure.
        let p_app =
            FluidState2::artificial_pressure(delta, Vec2::new(-2.0, 0.0), 1.0, 0.5, 1.0, 10.0, 1.0);
        // Separating -> no term.
        let p_sep =
            FluidState2::artificial_pressure(delta, Vec2::new(2.0, 0.0), 1.0, 0.5, 1.0, 10.0, 1.0);
        assert!(
            p_app > 0.0,
            "approaching pair needs a positive viscous term"
        );
        assert_eq!(p_sep, 0.0, "separating pair gets no viscous term");
        // alpha = 0 disables it entirely.
        let p0 =
            FluidState2::artificial_pressure(delta, Vec2::new(-2.0, 0.0), 1.0, 0.5, 1.0, 10.0, 0.0);
        assert_eq!(p0, 0.0);
    }

    #[test]
    fn test_fluid_cohesion_kernel() {
        let h = 0.5f32;
        assert!(FluidState2::cohesion(0.1, h) > FluidState2::cohesion(0.4, h));
        assert_eq!(FluidState2::cohesion(h, h), 0.0);
        assert_eq!(FluidState2::cohesion(0.9, h), 0.0);
    }

    #[test]
    fn test_fluid_cohesion_attracts() {
        // Two isolated particles with pressure disabled and strong
        // cohesion must accelerate toward each other.
        let mut world = crate::World2::new();
        world.gravity = Vec2::ZERO;
        let added = world
            .fluid
            .add_particles(&[Vec2::new(0.0, 0.0), Vec2::new(0.4, 0.0)]);
        assert_eq!(added, 2);
        world.fluid.config.stiffness = 0.0;
        world.fluid.config.viscosity = 0.0;
        world.fluid.config.surface_tension = 1000.0;
        world.step(1.0 / 60.0);
        let v0 = world.fluid.particles[0].velocity;
        let v1 = world.fluid.particles[1].velocity;
        assert!(v0.x > 0.0, "particle 0 not attracted toward +x: {v0:?}");
        assert!(v1.x < 0.0, "particle 1 not attracted toward -x: {v1:?}");
    }

    #[test]
    fn test_fluid_expansion_tank_stable() {
        // Tait EOS + Monaghan viscosity + surface tension, in the dam
        // tank: must stay finite, contained, and calm.
        let mut world = crate::World2::new();
        tank(&mut world);
        // Still-water tank (settling transient, hydrostatic support).
        lattice(&mut world, -0.5, -0.5, 12, 6, 0.25);
        world.fluid.config.gamma = 7.0;
        world.fluid.config.stiffness = 15.0;
        world.fluid.config.artificial_viscosity = 1.0;
        world.fluid.config.surface_tension = 0.5;
        let dt = 1.0 / 60.0;
        let mut peak = 0.0f32;
        for frame in 0..400 {
            world.step(dt);
            for p in world.fluid.particles.iter() {
                peak = peak.max(p.velocity.length());
            }
            if frame % 100 == 0 {
                finite_fluid(&world, "expansion tank");
            }
        }
        finite_fluid(&world, "expansion tank final");
        for p in world.fluid.particles.iter() {
            assert!(
                p.position.x > -1.0 && p.position.x < 5.0 && p.position.y > -0.6,
                "expansion particle escaped: p={}",
                p.position
            );
        }
        assert!(peak < 30.0, "expansion tank blew up, peak={peak}");
    }

    #[test]
    fn test_fluid_cfl_substeps() {
        let mut world = crate::World2::new();
        world
            .fluid
            .add_particles(&[Vec2::new(0.0, 0.0), Vec2::new(0.25, 0.0)]);
        // Disabled by default: always one pass.
        assert_eq!(world.fluid.cfl_substeps(1.0 / 60.0), 1);
        // Stiff Tait + a fast particle demands substeps.
        world.fluid.config.gamma = 7.0;
        world.fluid.config.stiffness = 200.0; // c0 ~ 37
        world.fluid.config.cfl = 0.4;
        world.fluid.config.max_substeps = 8;
        world.fluid.particles[0].velocity = Vec2::new(20.0, 0.0);
        let n = world.fluid.cfl_substeps(1.0 / 60.0);
        assert!((2..=8).contains(&n), "expected substeps, got {n}");
        // Capped by max_substeps.
        world.fluid.config.max_substeps = 2;
        assert_eq!(world.fluid.cfl_substeps(1.0 / 60.0), 2);
    }

    #[test]
    fn test_fluid_cfl_stabilizes_stiff_tait() {
        // A stiff Tait dam break that would violate CFL at a single pass
        // stays finite and contained once substepping is enabled.
        let mut world = crate::World2::new();
        tank(&mut world);
        lattice(&mut world, 0.0, 0.0, 8, 12, 0.25);
        world.fluid.config.gamma = 7.0;
        world.fluid.config.stiffness = 200.0;
        world.fluid.config.cfl = 0.4;
        world.fluid.config.max_substeps = 8;
        let mut peak = 0.0f32;
        for frame in 0..200 {
            world.step(1.0 / 60.0);
            for p in world.fluid.particles.iter() {
                peak = peak.max(p.velocity.length());
            }
            if frame % 50 == 0 {
                finite_fluid(&world, "cfl stiff tait");
            }
        }
        finite_fluid(&world, "cfl stiff tait final");
        for p in world.fluid.particles.iter() {
            assert!(
                p.position.x > -1.0 && p.position.x < 5.0 && p.position.y > -0.6,
                "CFL stiff particle escaped: p={}",
                p.position
            );
        }
        assert!(peak < 30.0, "CFL stiff run blew up, peak={peak}");
    }

    #[test]
    fn test_fluid_wind_drifts_floating_body() {
        // Ambient wind drives the Morison drag, sweeping a floating body.
        let mut world = crate::World2::new();
        tank(&mut world);
        lattice(&mut world, -0.5, -0.5, 12, 6, 0.25);
        for _ in 0..200 {
            world.step(1.0 / 60.0);
        }
        let b = world
            .add_dynamic_body(
                Vec2::new(2.0, 1.6),
                Shape2::Box {
                    half_extents: Vec2::new(0.3, 0.3),
                },
                0.5,
            )
            .expect("valid shape");
        let x0 = world.bodies[b].transform.position.x;
        world.ambient.drag.wind = Vec2::new(3.0, 0.0);
        for _ in 0..400 {
            world.step(1.0 / 60.0);
        }
        assert!(
            world.bodies[b].transform.position.x > x0 + 0.05,
            "wind did not drift the body: x0={x0} x={}",
            world.bodies[b].transform.position.x
        );
    }

    #[test]
    fn test_fluid_turbulence_deterministic() {
        fn run() -> Vec2 {
            let mut world = crate::World2::new();
            tank(&mut world);
            lattice(&mut world, 0.0, 0.0, 6, 8, 0.25);
            world.ambient.turbulence.enabled = true;
            world.ambient.turbulence.amplitude = 5.0;
            world.ambient.turbulence.frequency = 0.5;
            world.ambient.turbulence.seed = 7;
            world.ambient.turbulence.octaves = 3;
            for _ in 0..120 {
                world.step(1.0 / 60.0);
            }
            world.fluid.particles[0].velocity
        }
        let a = run();
        let b = run();
        assert_eq!(a.x.to_bits(), b.x.to_bits());
        assert_eq!(a.y.to_bits(), b.y.to_bits());
    }

    #[test]
    fn test_fluid_adaptive_resolution_conserves_mass() {
        let mut world = crate::World2::new();
        tank(&mut world);
        // Calm still water (adaptive resolution targets resolved regions,
        // not the violent surface of a dam break).
        lattice(&mut world, -0.5, -0.5, 12, 6, 0.25);
        world.fluid.config.adaptive_resolution = true;
        let total0: f32 = world.fluid.particles.iter().map(|p| p.mass).sum();
        for frame in 0..200 {
            world.step(1.0 / 60.0);
            if frame % 50 == 0 {
                finite_fluid(&world, "adaptive");
            }
        }
        finite_fluid(&world, "adaptive final");
        let total1: f32 = world.fluid.particles.iter().map(|p| p.mass).sum();
        assert!(
            (total1 - total0).abs() < 1e-3,
            "adaptive resolution lost mass: {total0} -> {total1}"
        );
        // Count stays within the cap and particles stay contained/calm.
        assert!(world.fluid.particles.len() <= MAX_PARTICLES_2D);
        for p in world.fluid.particles.iter() {
            assert!(
                p.position.x > -1.0 && p.position.x < 5.0 && p.position.y > -0.6,
                "adaptive particle escaped: p={}",
                p.position
            );
        }
    }

    #[test]
    fn test_fluid_split_conserves_mass_and_caps() {
        // Directly exercise the split path on under-dense particles.
        let mut world = crate::World2::new();
        let added = world.fluid.add_particles(&[Vec2::new(0.0, 0.0)]);
        assert_eq!(added, 1);
        for p in world.fluid.particles.iter_mut() {
            p.density = 0.1; // well under the 0.5 split threshold
        }
        let before: f32 = world.fluid.particles.iter().map(|p| p.mass).sum();
        world.fluid.resample(0.5);
        assert_eq!(
            world.fluid.particles.len(),
            2,
            "under-dense particle should split"
        );
        let after: f32 = world.fluid.particles.iter().map(|p| p.mass).sum();
        assert!((after - before).abs() < 1e-6, "split must conserve mass");
    }

    #[test]
    fn test_fluid_boundary_particles_present() {
        let mut world = crate::World2::new();
        tank(&mut world);
        lattice(&mut world, 0.0, 0.0, 6, 8, 0.25);
        world.fluid.config.boundary_particles = true;
        world.step(1.0 / 60.0);
        assert!(
            !world.fluid.boundary.is_empty(),
            "static bodies should generate boundary particles"
        );
        // Deterministic: two identical worlds generate identical ghosts.
        let mut other = crate::World2::new();
        tank(&mut other);
        lattice(&mut other, 0.0, 0.0, 6, 8, 0.25);
        other.fluid.config.boundary_particles = true;
        other.step(1.0 / 60.0);
        assert_eq!(
            world.fluid.boundary.len(),
            other.fluid.boundary.len(),
            "boundary count differs: {} vs {}",
            world.fluid.boundary.len(),
            other.fluid.boundary.len()
        );
        for (a, b) in world.fluid.boundary.iter().zip(other.fluid.boundary.iter()) {
            assert_eq!(a.position.x.to_bits(), b.position.x.to_bits());
            assert_eq!(a.position.y.to_bits(), b.position.y.to_bits());
        }
    }

    #[test]
    fn test_fluid_boundary_particles_contained() {
        // Dam tank with boundary pressure: stays finite, contained, calm.
        let mut world = crate::World2::new();
        tank(&mut world);
        lattice(&mut world, 0.0, 0.0, 8, 12, 0.25);
        world.fluid.config.boundary_particles = true;
        let mut peak = 0.0f32;
        for frame in 0..300 {
            world.step(1.0 / 60.0);
            for p in world.fluid.particles.iter() {
                peak = peak.max(p.velocity.length());
            }
            if frame % 100 == 0 {
                finite_fluid(&world, "boundary");
            }
        }
        finite_fluid(&world, "boundary final");
        for p in world.fluid.particles.iter() {
            assert!(
                p.position.x > -1.0 && p.position.x < 5.0 && p.position.y > -0.6,
                "boundary particle escaped: p={}",
                p.position
            );
        }
        assert!(peak < 30.0, "boundary run blew up, peak={peak}");
    }
}
