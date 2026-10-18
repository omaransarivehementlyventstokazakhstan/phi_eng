//! Continuous collision detection (CCD) via adaptive substepping.
//!
//! When enabled, [`World2::step`](crate::World2::step) /
//! [`World3::step`](crate::World3::step) measure the fastest effective
//! body speed (linear plus angular sweep) and split the step into enough
//! substeps that no body travels more than [`CcdConfig::max_movement`]
//! per substep. Each substep runs the full pipeline (gravity, ambient,
//! fluid, broadphase, narrowphase, solve, integrate), so fast bodies can
//! no longer tunnel through thin walls within a single frame.
//!
//! The default config is a complete no-op (one substep, bit-identical
//! trajectories). Non-default configs persist through snapshots at
//! schema v6 and cross the FFI bridge.

/// CCD configuration, shared by 2D and 3D worlds. Dimension-agnostic:
/// only speeds, distances, and counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CcdConfig {
    /// Master switch. `false` (default) = single substep, legacy behavior.
    pub enabled: bool,
    /// Maximum distance any body may travel within one substep. Must be
    /// smaller than the thinnest collider the scene relies on; the
    /// default (`0.1`) suits decimeter-scale scenes at 60 Hz.
    pub max_movement: f32,
    /// Hard cap on substeps per world step (cost guard). Must be `>= 1`.
    pub max_substeps: u32,
}

impl Default for CcdConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_movement: 0.1,
            max_substeps: 8,
        }
    }
}

impl CcdConfig {
    /// True for the default (legacy, single-substep) configuration.
    /// Snapshot schema selection uses this: default configs never
    /// trigger a schema bump.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// True when substepping can actually engage.
    pub fn is_active(&self) -> bool {
        self.enabled
            && self.max_substeps > 1
            && self.max_movement.is_finite()
            && self.max_movement > 0.0
    }

    /// Number of substeps for a step of length `dt` given the peak
    /// effective body speed. Always `1` when inactive, on non-positive
    /// `dt`, or on non-finite/non-positive speed.
    pub fn substeps(&self, dt: f32, peak_speed: f32) -> u32 {
        if !self.is_active() || dt <= 0.0 || !peak_speed.is_finite() || peak_speed <= 0.0 {
            return 1;
        }
        let max = self.max_substeps.max(1);
        let need = (peak_speed * dt / self.max_movement).ceil() as i64;
        need.clamp(1, max as i64) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::CcdConfig;

    #[test]
    fn ccd_config_defaults_single_step() {
        let cfg = CcdConfig::default();
        assert!(cfg.is_default());
        assert!(!cfg.is_active());
        assert_eq!(cfg.substeps(1.0 / 60.0, 100.0), 1);
    }

    #[test]
    fn ccd_substep_count_scales_with_speed() {
        let cfg = CcdConfig {
            enabled: true,
            max_movement: 0.1,
            max_substeps: 8,
        };
        assert!(cfg.is_active());
        // 5 units/s at 60 Hz = 0.083/step < 0.1 -> exactly 1 substep.
        assert_eq!(cfg.substeps(1.0 / 60.0, 5.0), 1);
        // 11 units/s -> 0.183/step -> 2 substeps.
        assert_eq!(cfg.substeps(1.0 / 60.0, 11.0), 2);
        // Extreme speed clamps at the cap instead of exploding cost.
        assert_eq!(cfg.substeps(1.0 / 60.0, 1.0e6), 8);
    }

    #[test]
    fn ccd_rejects_degenerate_inputs() {
        let cfg = CcdConfig {
            enabled: true,
            max_movement: 0.1,
            max_substeps: 8,
        };
        assert_eq!(cfg.substeps(0.0, 10.0), 1);
        assert_eq!(cfg.substeps(-1.0, 10.0), 1);
        assert_eq!(cfg.substeps(1.0 / 60.0, f32::NAN), 1);
        assert_eq!(cfg.substeps(1.0 / 60.0, f32::INFINITY), 1);
        let off = CcdConfig {
            max_movement: 0.0,
            ..cfg
        };
        assert!(!off.is_active());
        let one = CcdConfig {
            max_substeps: 1,
            ..cfg
        };
        assert!(!one.is_active());
    }
}
