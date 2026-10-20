//! Nuclear physics: radioactive decay, neutron activation, fission and
//! chain reactions.
//!
//! Position in [`crate::World2::step`]: the nuclear phase runs *before*
//! the thermal phase. It fills a per-body heat budget (the energy
//! released by decay and fission) and a per-body ionizing-radiation
//! budget, which the thermal phase then turns into temperature. Running
//! it first means the thermal radiation sees the heat in the same step.
//!
//! # Determinism
//!
//! The decay uses `exp`, which is **not** portable across platforms (the
//! libm implementations disagree in the last ULP). It goes through
//! [`phieng_math::exponential::exp`], the engine's deterministic
//! exponential, so a half-life drifts by the same amount everywhere and
//! Rust and the managed C# mirror agree bit-for-bit.
//!
//! # Model
//!
//! The model is game-scale rather than reactor-scale: it is built from
//! rates, not from particle transport, so it is O(n) per step and
//! deterministic without an RNG.
//!
//! * **Decay**: a nuclide with a half-life `T` has the decay constant
//!   `lambda = ln(2) / T`; its activity falls as `exp(-lambda * dt)`
//!   and it releases `activity * decay_energy * dt` joules per step.
//!   A fraction `ionization_fraction` of that leaves as ionizing
//!   radiation; the rest becomes heat.
//! * **Activation**: a nuclide with a non-zero daughter turns into that
//!   daughter when it captures a neutron.
//! * **Fission**: a fissile nuclide with fuel left absorbs neutrons
//!   from the ambient flux and from its neighbours. Each fission
//!   consumes fuel, releases `fission_energy` joules and produces
//!   `neutron_yield` neutrons. A fraction of those are *delayed* (they
//!   leak out over `delayed_neutron_delay` seconds), which is what
//!   makes a chain reaction prompt-critical rather than immediately
//!   divergent.
//!
//! Neutrons travel by a simple nearest-neighbour spread: a body's free
//! neutrons are shared equally among the bodies within
//! `neutron_range`. That is a diffusion approximation, not a transport
//! solve; see `LIMITATIONS.md`.

use super::body::{RigidBody2, RigidBody3};
use phieng_math::exponential::exp;

/// The per-body nuclear state.
///
/// This lives on the body (like the thermal state) so that it follows
/// the body through removals and so that the hot part of the body
/// struct is untouched: the nuclear fields are cold and sit at the
/// tail.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NuclearState {
    /// The nuclide index. Zero means "no nuclide": the body is not
    /// radioactive and takes no part in the nuclear phase.
    pub isotope: u16,
    /// The activity, in decays per second.
    pub activity: f32,
    /// The remaining fissile fuel, 0..=1.
    pub fuel: f32,
    /// The free neutron population.
    pub neutrons: f32,
    /// The delayed neutron population.
    pub delayed_neutrons: f32,
    /// The accumulated ionizing dose.
    pub dose: f32,
    /// The accumulated neutron-capture progress. A body with a daughter
    /// nuclide captures a neutron when this reaches one, and then becomes
    /// that daughter.
    pub capture_progress: f32,
}

/// A nuclide the engine knows about.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Isotope {
    /// The half-life in seconds. Zero or less means the nuclide is
    /// stable: it never decays.
    pub half_life: f32,
    /// The energy released per decay, in joules.
    pub decay_energy: f32,
    /// The fraction of the decay energy that leaves as ionizing
    /// radiation; the rest becomes heat. Clamped to 0..=1.
    pub ionization_fraction: f32,
    /// The nuclide produced by neutron capture; zero means the capture
    /// leaves the nuclide unchanged.
    pub daughter: u16,
    /// The neutron absorption cross-section, in barns
    /// (1 barn = 1e-28 m^2).
    pub absorption_cross_section: f32,
    /// The fission cross-section, in barns (1 barn = 1e-28 m^2).
    ///
    /// Used as a relative weight - see
    /// [`Isotope::absorption_cross_section`].
    pub fission_cross_section: f32,
    /// The neutrons released per fission.
    pub neutron_yield: f32,
    /// Whether the nuclide can fission.
    pub is_fissile: bool,
}

impl Default for Isotope {
    fn default() -> Self {
        Self {
            half_life: 0.0,
            decay_energy: 0.0,
            ionization_fraction: 0.0,
            daughter: 0,
            absorption_cross_section: 0.0,
            fission_cross_section: 0.0,
            neutron_yield: 0.0,
            is_fissile: false,
        }
    }
}

impl Isotope {
    /// The decay constant `lambda = ln(2) / half_life`, in 1/s.
    ///
    /// A stable nuclide (half-life <= 0) has the constant zero.
    pub fn decay_constant(&self) -> f32 {
        if self.half_life <= 0.0 || !self.half_life.is_finite() {
            return 0.0;
        }
        LN2 / self.half_life
    }
}

/// `ln(2)`, rounded to `f32`.
///
/// Computed once with the engine's deterministic logarithm so that the
/// Rust core and the managed C# mirror agree bit-for-bit.
const LN2: f32 = std::f32::consts::LN_2;

/// The nuclear simulation settings.
#[derive(Debug, Clone, PartialEq)]
pub struct NuclearConfig {
    /// Master switch. Off by default, so a default world is
    /// bit-identical to a pre-nuclear one.
    pub enabled: bool,
    /// The nuclide table. Index zero is the "no nuclide" placeholder;
    /// the real nuclides start at index one.
    pub isotopes: Vec<Isotope>,
    /// Nuclear seconds per simulation second. A half-life of years
    /// becomes visible with a large value (10^5 turns a 5730-year
    /// carbon-14 half-life into half a minute of simulation).
    pub time_scale: f32,
    /// The energy released per fission, in joules.
    pub fission_energy: f32,
    /// The background neutron flux (a dimensionless rate).
    pub ambient_flux: f32,
    /// The fission rate constant: how fast a unit flux over a unit
    /// cross-section fissions a unit of fuel.
    pub fission_rate: f32,
    /// The capture rate constant: how fast a unit flux over a unit
    /// absorption cross-section captures a neutron (and turns the
    /// nuclide into its daughter).
    pub activation_rate: f32,
    /// The fuel consumed per fission.
    pub fuel_per_fission: f32,
    /// The range within which free neutrons are shared, in metres.
    pub neutron_range: f32,
    /// The range within which ionizing radiation is deposited, in
    /// metres. Radiation that reaches no body is simply lost.
    pub dose_range: f32,
    /// The fraction of the fission neutrons that are delayed.
    pub delayed_neutron_fraction: f32,
    /// The delay of the delayed neutrons, in nuclear seconds.
    pub delayed_neutron_delay: f32,
    /// The maximum activity, in decays per second.
    pub max_activity: f32,
    /// The dose accumulated per joule of ionizing radiation.
    pub dose_per_joule: f32,
    /// The maximum accumulated dose.
    pub max_dose: f32,
}

impl Default for NuclearConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            isotopes: vec![Isotope::default()],
            time_scale: 1.0,
            fission_energy: 3.204e-11, // 200 MeV
            ambient_flux: 0.0,
            fission_rate: 1.0,
            activation_rate: 1.0,
            fuel_per_fission: 1e-6,
            neutron_range: 1.0,
            dose_range: 1.0,
            delayed_neutron_fraction: 0.006,
            delayed_neutron_delay: 0.5,
            max_activity: 1e30,
            dose_per_joule: 1.0,
            max_dose: 1e30,
        }
    }
}

impl NuclearConfig {
    /// True when the nuclear phase should run.
    pub fn is_active(&self) -> bool {
        self.enabled
            && self.time_scale.is_finite()
            && self.time_scale > 0.0
            && self.isotopes.len() > 1
    }

    /// The nuclide with the given index, if it exists.
    pub fn isotope(&self, index: u16) -> Option<&Isotope> {
        self.isotopes.get(index as usize)
    }
}

/// The per-step nuclear scratch.
///
/// Kept separate from the configuration so that the phase can borrow
/// the configuration immutably and the scratch mutably at the same time
/// (the configuration owns the nuclide table, so it is not `Copy`).
#[derive(Debug, Clone, Default)]
pub struct NuclearScratch {
    /// The heat produced by nuclear processes this step, in joules.
    heat: Vec<f32>,
    /// The ionizing radiation produced this step, in joules.
    radiation: Vec<f32>,
    /// The neutron flux arriving at each body this step.
    flux: Vec<f32>,
    /// The dose weight of each body this step.
    dose_weights: Vec<f32>,
}

impl NuclearScratch {
    /// The heat produced at a body this step, in joules.
    pub fn heat_at(&self, index: usize) -> f32 {
        self.heat.get(index).copied().unwrap_or(0.0)
    }

    /// The ionizing radiation produced at a body this step, in joules.
    pub fn radiation_at(&self, index: usize) -> f32 {
        self.radiation.get(index).copied().unwrap_or(0.0)
    }
}

/// The per-world nuclear state and configuration.
#[derive(Debug, Clone, Default)]
pub struct NuclearState2 {
    /// The nuclear configuration.
    pub config: NuclearConfig,
    /// The per-step scratch.
    pub scratch: NuclearScratch,
}

impl NuclearState2 {
    /// A fresh state with the default configuration.
    pub fn new() -> Self {
        Self::default()
    }
}

/// The minimum of two values, ignoring NaN.
///
/// This is Rust's `f32::min` (IEEE `minNum`), spelled out so the
/// managed mirror can match it exactly: .NET's `Math.Min` *propagates*
/// NaN instead, which would make the two engines disagree.
#[inline]
fn min_num(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else {
        a.min(b)
    }
}

/// The maximum of two values, ignoring NaN.
///
/// The NaN-ignoring counterpart of [`min_num`]; .NET's `Math.Max`
/// propagates NaN, which would make the two engines disagree.
#[inline]
fn max_num(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else {
        a.max(b)
    }
}

/// The product of two values, saturated at `f32::MAX` instead of
/// overflowing to infinity.
///
/// An infinite energy would turn into NaN in the heat/ionizing split
/// (`infinity * 0.0`), and a NaN temperature poisons the body for good.
#[inline]
fn saturating_product(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return f32::NAN;
    }
    let product = a * b;
    if product.is_infinite() {
        if (a < 0.0) != (b < 0.0) {
            f32::NEG_INFINITY
        } else {
            f32::MAX
        }
    } else {
        product
    }
}

/// The decay constant of a half-life, in 1/s.
pub fn decay_constant(half_life: f32) -> f32 {
    if half_life <= 0.0 || !half_life.is_finite() {
        return 0.0;
    }
    LN2 / half_life
}

/// Run the nuclear phase for a 2D world.
///
/// Fills the world's heat and radiation budgets. The temperature is
/// updated by [`apply_nuclear_heat_2d`], which the world calls right
/// after this.
pub fn step_nuclear_2d(
    bodies: &mut [RigidBody2],
    config: &NuclearConfig,
    scratch: &mut NuclearScratch,
    dt: f32,
) {
    // The scratch is cleared *before* the early returns: the world
    // applies the heat budget after this call, and a stale budget would
    // be re-applied every step (an unbounded temperature ramp).
    let n = bodies.len();
    if scratch.heat.len() != n {
        scratch.heat.resize(n, 0.0);
        scratch.radiation.resize(n, 0.0);
        scratch.flux.resize(n, 0.0);
    }
    for v in scratch.heat.iter_mut() {
        *v = 0.0;
    }
    for v in scratch.radiation.iter_mut() {
        *v = 0.0;
    }
    for v in scratch.flux.iter_mut() {
        *v = 0.0;
    }

    if !config.is_active() || dt <= 0.0 || bodies.is_empty() {
        return;
    }
    let dt_nuclear = dt * config.time_scale;
    if dt_nuclear <= 0.0 {
        return;
    }

    // 1. Decay: the activity falls and releases energy.
    for (i, body) in bodies.iter_mut().enumerate() {
        let ns = &mut body.nuclear;
        if ns.isotope == 0 || ns.activity <= 0.0 {
            continue;
        }
        let Some(iso) = config.isotope(ns.isotope) else {
            continue;
        };
        let lambda = iso.decay_constant();
        if lambda <= 0.0 {
            // A stable nuclide never decays.
            ns.activity = 0.0;
            continue;
        }
        if iso.decay_energy <= 0.0 {
            continue;
        }

        // The energy released in this step uses the start-of-step
        // activity (explicit, like the thermal phase).
        //
        // The product can overflow to infinity for extreme inputs, and
        // `infinity * 0.0` is NaN - which would poison the body's
        // temperature for good. The split is therefore computed on a
        // saturated energy and with an explicit zero-ionizing branch.
        let energy = saturating_product(ns.activity, iso.decay_energy) * dt_nuclear;
        let fraction = iso.ionization_fraction.clamp(0.0, 1.0);
        let ionizing = if fraction <= 0.0 {
            0.0
        } else {
            energy * fraction
        };
        scratch.heat[i] += energy - ionizing;
        scratch.radiation[i] += ionizing;

        ns.activity *= exp(-lambda * dt_nuclear);
        if !ns.activity.is_finite() || ns.activity < 0.0 {
            ns.activity = 0.0;
        }
        ns.activity = min_num(ns.activity, config.max_activity);
    }

    // 2. Fission and the chain reaction.
    if config.fission_rate > 0.0 {
        step_fission_2d(bodies, config, scratch, dt_nuclear);
    }

    // 3. Deposit the ionizing radiation into the nearby bodies.
    deposit_dose_2d(bodies, config, scratch);
}

/// The fission and chain-reaction step (2D).
fn step_fission_2d(
    bodies: &mut [RigidBody2],
    config: &NuclearConfig,
    scratch: &mut NuclearScratch,
    dt_nuclear: f32,
) {
    let range_sq = config.neutron_range * config.neutron_range;

    // 1. Spread the free neutrons to the neighbours within the range.
    //    Read-only, so the bodies can be borrowed immutably here.
    for i in 0..bodies.len() {
        let available = bodies[i].nuclear.neutrons;
        if available <= 0.0 {
            continue;
        }
        let origin = bodies[i].transform.position;
        let mut neighbours = 0u32;
        for (j, other) in bodies.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = origin - other.transform.position;
            if d.length_squared() <= range_sq {
                neighbours += 1;
            }
        }
        if neighbours == 0 {
            // The neutrons escape.
            continue;
        }
        let share = available / neighbours as f32;
        for (j, other) in bodies.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = origin - other.transform.position;
            if d.length_squared() <= range_sq {
                scratch.flux[j] += share;
            }
        }
    }
    // The neutrons have been handed over (or escaped).
    for body in bodies.iter_mut() {
        body.nuclear.neutrons = 0.0;
    }

    // 2. The delayed neutrons leak out over the delay and join the free
    //    pool.
    for body in bodies.iter_mut() {
        let ns = &mut body.nuclear;
        if ns.delayed_neutrons <= 0.0 {
            continue;
        }
        // A non-positive delay means "release everything now". The
        // explicit branch avoids relying on `dt / f32::MIN_POSITIVE`
        // overflowing to -inf.
        let released = if config.delayed_neutron_delay > 0.0 {
            ns.delayed_neutrons * (1.0 - exp(-dt_nuclear / config.delayed_neutron_delay))
        } else {
            ns.delayed_neutrons
        };
        ns.delayed_neutrons -= released;
        ns.neutrons += released;
    }

    // 3. The fissions.
    for (i, body) in bodies.iter_mut().enumerate() {
        let ns = &mut body.nuclear;
        if ns.isotope == 0 || ns.fuel <= 0.0 {
            continue;
        }
        let Some(iso) = config.isotope(ns.isotope) else {
            continue;
        };
        if !iso.is_fissile || iso.fission_cross_section <= 0.0 {
            continue;
        }

        let flux = config.ambient_flux + scratch.flux[i];
        if flux <= 0.0 {
            continue;
        }
        let rate = flux * iso.fission_cross_section * ns.fuel * config.fission_rate;
        let mut fissions = rate * dt_nuclear;
        if fissions <= 0.0 {
            continue;
        }
        // The fissions cannot exceed the fuel actually present: without
        // this cap a body would keep producing heat and neutrons long
        // after its fuel ran out.
        if config.fuel_per_fission > 0.0 {
            fissions = fissions.min(ns.fuel / config.fuel_per_fission);
        }
        if fissions <= 0.0 {
            continue;
        }

        ns.fuel = (ns.fuel - fissions * config.fuel_per_fission).max(0.0);

        let energy = saturating_product(fissions, config.fission_energy);
        scratch.heat[i] += energy;

        let neutrons = fissions * iso.neutron_yield;
        let delayed = neutrons * config.delayed_neutron_fraction.clamp(0.0, 1.0);
        ns.delayed_neutrons += delayed;
        ns.neutrons += neutrons - delayed;
    }

    // 4. Capture: a body with a daughter nuclide absorbs the neutrons
    //    in its flux and becomes that daughter. The capture is
    //    mean-field (a rate, not a per-neutron coin flip), so it stays
    //    deterministic without an RNG.
    for (i, body) in bodies.iter_mut().enumerate() {
        let ns = &mut body.nuclear;
        if ns.isotope == 0 {
            continue;
        }
        let Some(iso) = config.isotope(ns.isotope) else {
            continue;
        };
        if iso.daughter == 0 || iso.absorption_cross_section <= 0.0 {
            continue;
        }
        let flux = config.ambient_flux + scratch.flux[i];
        if flux <= 0.0 {
            continue;
        }
        let rate = flux * iso.absorption_cross_section * config.activation_rate;
        if rate <= 0.0 {
            continue;
        }
        ns.capture_progress += rate * dt_nuclear;
        if ns.capture_progress >= 1.0 {
            ns.isotope = iso.daughter;
            ns.capture_progress = 0.0;
            ns.activity = 0.0;
            ns.fuel = 0.0;
        }
    }
}

/// Run the nuclear phase for a 3D world.
pub fn step_nuclear_3d(
    bodies: &mut [RigidBody3],
    config: &NuclearConfig,
    scratch: &mut NuclearScratch,
    dt: f32,
) {
    // The scratch is cleared *before* the early returns: the world
    // applies the heat budget after this call, and a stale budget would
    // be re-applied every step (an unbounded temperature ramp).
    let n = bodies.len();
    if scratch.heat.len() != n {
        scratch.heat.resize(n, 0.0);
        scratch.radiation.resize(n, 0.0);
        scratch.flux.resize(n, 0.0);
    }
    for v in scratch.heat.iter_mut() {
        *v = 0.0;
    }
    for v in scratch.radiation.iter_mut() {
        *v = 0.0;
    }
    for v in scratch.flux.iter_mut() {
        *v = 0.0;
    }

    if !config.is_active() || dt <= 0.0 || bodies.is_empty() {
        return;
    }
    let dt_nuclear = dt * config.time_scale;
    if dt_nuclear <= 0.0 {
        return;
    }

    for (i, body) in bodies.iter_mut().enumerate() {
        let ns = &mut body.nuclear;
        if ns.isotope == 0 || ns.activity <= 0.0 {
            continue;
        }
        let Some(iso) = config.isotope(ns.isotope) else {
            continue;
        };
        let lambda = iso.decay_constant();
        if lambda <= 0.0 {
            ns.activity = 0.0;
            continue;
        }
        if iso.decay_energy <= 0.0 {
            continue;
        }
        let energy = saturating_product(ns.activity, iso.decay_energy) * dt_nuclear;
        let fraction = iso.ionization_fraction.clamp(0.0, 1.0);
        let ionizing = if fraction <= 0.0 {
            0.0
        } else {
            energy * fraction
        };
        scratch.heat[i] += energy - ionizing;
        scratch.radiation[i] += ionizing;

        ns.activity *= exp(-lambda * dt_nuclear);
        if !ns.activity.is_finite() || ns.activity < 0.0 {
            ns.activity = 0.0;
        }
        ns.activity = min_num(ns.activity, config.max_activity);
    }

    // 2. Fission and the chain reaction.
    if config.fission_rate > 0.0 {
        step_fission_3d(bodies, config, scratch, dt_nuclear);
    }

    deposit_dose_3d(bodies, config, scratch);
}

/// The fission and chain-reaction step (3D).
fn step_fission_3d(
    bodies: &mut [RigidBody3],
    config: &NuclearConfig,
    scratch: &mut NuclearScratch,
    dt_nuclear: f32,
) {
    let range_sq = config.neutron_range * config.neutron_range;

    // 1. Spread the free neutrons to the neighbours within the range.
    for i in 0..bodies.len() {
        let available = bodies[i].nuclear.neutrons;
        if available <= 0.0 {
            continue;
        }
        let origin = bodies[i].transform.position;
        let mut neighbours = 0u32;
        for (j, other) in bodies.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = origin - other.transform.position;
            if d.length_squared() <= range_sq {
                neighbours += 1;
            }
        }
        if neighbours == 0 {
            continue;
        }
        let share = available / neighbours as f32;
        for (j, other) in bodies.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = origin - other.transform.position;
            if d.length_squared() <= range_sq {
                scratch.flux[j] += share;
            }
        }
    }
    for body in bodies.iter_mut() {
        body.nuclear.neutrons = 0.0;
    }

    // 2. The delayed neutrons leak out over the delay.
    for body in bodies.iter_mut() {
        let ns = &mut body.nuclear;
        if ns.delayed_neutrons <= 0.0 {
            continue;
        }
        // A non-positive delay means "release everything now". The
        // explicit branch avoids relying on `dt / f32::MIN_POSITIVE`
        // overflowing to -inf.
        let released = if config.delayed_neutron_delay > 0.0 {
            ns.delayed_neutrons * (1.0 - exp(-dt_nuclear / config.delayed_neutron_delay))
        } else {
            ns.delayed_neutrons
        };
        ns.delayed_neutrons -= released;
        ns.neutrons += released;
    }

    // 3. The fissions.
    for (i, body) in bodies.iter_mut().enumerate() {
        let ns = &mut body.nuclear;
        if ns.isotope == 0 || ns.fuel <= 0.0 {
            continue;
        }
        let Some(iso) = config.isotope(ns.isotope) else {
            continue;
        };
        if !iso.is_fissile || iso.fission_cross_section <= 0.0 {
            continue;
        }
        let flux = config.ambient_flux + scratch.flux[i];
        if flux <= 0.0 {
            continue;
        }
        let rate = flux * iso.fission_cross_section * ns.fuel * config.fission_rate;
        let mut fissions = rate * dt_nuclear;
        if fissions <= 0.0 {
            continue;
        }
        // The fissions cannot exceed the fuel actually present.
        if config.fuel_per_fission > 0.0 {
            fissions = fissions.min(ns.fuel / config.fuel_per_fission);
        }
        if fissions <= 0.0 {
            continue;
        }
        ns.fuel = (ns.fuel - fissions * config.fuel_per_fission).max(0.0);
        scratch.heat[i] += saturating_product(fissions, config.fission_energy);
        let neutrons = fissions * iso.neutron_yield;
        let delayed = neutrons * config.delayed_neutron_fraction.clamp(0.0, 1.0);
        ns.delayed_neutrons += delayed;
        ns.neutrons += neutrons - delayed;
    }

    // 4. Capture (see the 2D counterpart).
    for (i, body) in bodies.iter_mut().enumerate() {
        let ns = &mut body.nuclear;
        if ns.isotope == 0 {
            continue;
        }
        let Some(iso) = config.isotope(ns.isotope) else {
            continue;
        };
        if iso.daughter == 0 || iso.absorption_cross_section <= 0.0 {
            continue;
        }
        let flux = config.ambient_flux + scratch.flux[i];
        if flux <= 0.0 {
            continue;
        }
        let rate = flux * iso.absorption_cross_section * config.activation_rate;
        if rate <= 0.0 {
            continue;
        }
        ns.capture_progress += rate * dt_nuclear;
        if ns.capture_progress >= 1.0 {
            ns.isotope = iso.daughter;
            ns.capture_progress = 0.0;
            ns.activity = 0.0;
            ns.fuel = 0.0;
        }
    }
}

/// Deposit the ionizing radiation into the nearby bodies (2D).
///
/// The energy is shared out with an inverse-square weight, so the total
/// deposited energy never exceeds the energy produced: radiation that
/// reaches no body is simply lost (it left the system).
fn deposit_dose_2d(
    bodies: &mut [RigidBody2],
    config: &NuclearConfig,
    scratch: &mut NuclearScratch,
) {
    let range_sq = config.dose_range * config.dose_range;
    let n = bodies.len();
    if scratch.dose_weights.len() != n {
        scratch.dose_weights.resize(n, 0.0);
    }
    for w in scratch.dose_weights.iter_mut() {
        *w = 0.0;
    }

    // The weights: an inverse-square falloff over the bodies in range.
    for i in 0..n {
        let energy = scratch.radiation[i];
        if energy <= 0.0 {
            continue;
        }
        let origin = bodies[i].transform.position;
        let mut total = 0.0f32;
        for (j, other) in bodies.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = origin - other.transform.position;
            let d2 = d.length_squared();
            if d2 > range_sq {
                continue;
            }
            let w = 1.0 / max_num(d2, 1e-6);
            scratch.dose_weights[j] = w;
            total += w;
        }
        if total <= 0.0 {
            // The radiation reached nobody.
            continue;
        }
        let dose = energy * config.dose_per_joule;
        for (j, other) in bodies.iter_mut().enumerate() {
            if i == j || scratch.dose_weights[j] <= 0.0 {
                continue;
            }
            let share = scratch.dose_weights[j] / total;
            let ns = &mut other.nuclear;
            ns.dose += dose * share;
            if !ns.dose.is_finite() || ns.dose < 0.0 {
                ns.dose = 0.0;
            }
            ns.dose = min_num(ns.dose, config.max_dose);
        }
        for w in scratch.dose_weights.iter_mut() {
            *w = 0.0;
        }
    }
}

/// Deposit the ionizing radiation into the nearby bodies (3D).
fn deposit_dose_3d(
    bodies: &mut [RigidBody3],
    config: &NuclearConfig,
    scratch: &mut NuclearScratch,
) {
    let range_sq = config.dose_range * config.dose_range;
    let n = bodies.len();
    if scratch.dose_weights.len() != n {
        scratch.dose_weights.resize(n, 0.0);
    }
    for w in scratch.dose_weights.iter_mut() {
        *w = 0.0;
    }

    for i in 0..n {
        let energy = scratch.radiation[i];
        if energy <= 0.0 {
            continue;
        }
        let origin = bodies[i].transform.position;
        let mut total = 0.0f32;
        for (j, other) in bodies.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = origin - other.transform.position;
            let d2 = d.length_squared();
            if d2 > range_sq {
                continue;
            }
            let w = 1.0 / max_num(d2, 1e-6);
            scratch.dose_weights[j] = w;
            total += w;
        }
        if total <= 0.0 {
            continue;
        }
        let dose = energy * config.dose_per_joule;
        for (j, other) in bodies.iter_mut().enumerate() {
            if i == j || scratch.dose_weights[j] <= 0.0 {
                continue;
            }
            let share = scratch.dose_weights[j] / total;
            let ns = &mut other.nuclear;
            ns.dose += dose * share;
            if !ns.dose.is_finite() || ns.dose < 0.0 {
                ns.dose = 0.0;
            }
            ns.dose = min_num(ns.dose, config.max_dose);
        }
        for w in scratch.dose_weights.iter_mut() {
            *w = 0.0;
        }
    }
}

/// Apply the nuclear heat to the body temperatures (2D).
///
/// The temperature clamps come from the thermal configuration, so the
/// absolute bounds are the same ones the thermal phase enforces. The
/// rate clamp is deliberately *not* applied: a nuclear source has a
/// physical rate that is not an integration artefact.
pub fn apply_nuclear_heat_2d(
    bodies: &mut [RigidBody2],
    scratch: &NuclearScratch,
    thermal_config: &super::thermal::ThermalConfig2D,
) {
    for (i, body) in bodies.iter_mut().enumerate() {
        let heat = scratch.heat_at(i);
        if heat <= 0.0 {
            continue;
        }
        if body.is_static {
            // Statics are infinite-capacity reservoirs.
            continue;
        }
        let capacity = body.mass * body.material.specific_heat;
        if capacity <= 0.0 {
            continue;
        }
        let delta = heat / capacity;
        body.temperature = (body.temperature + delta).clamp(
            thermal_config.min_temperature,
            thermal_config.max_temperature,
        );
    }
}

/// Apply the nuclear heat to the body temperatures (3D).
pub fn apply_nuclear_heat_3d(
    bodies: &mut [RigidBody3],
    scratch: &NuclearScratch,
    thermal_config: &super::thermal::ThermalConfig3D,
) {
    for (i, body) in bodies.iter_mut().enumerate() {
        let heat = scratch.heat_at(i);
        if heat <= 0.0 {
            continue;
        }
        if body.is_static {
            continue;
        }
        let capacity = body.mass * body.material.specific_heat;
        if capacity <= 0.0 {
            continue;
        }
        let delta = heat / capacity;
        body.temperature = (body.temperature + delta).clamp(
            thermal_config.min_temperature,
            thermal_config.max_temperature,
        );
    }
}
