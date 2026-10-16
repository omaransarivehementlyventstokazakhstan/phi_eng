// Validation uses Vec3 from glam through the body types; no direct import needed.

/// Validation configuration controlling runtime checks.
#[derive(Debug, Clone, Copy)]
pub struct ValidationConfig {
    /// Check for NaN/Inf in positions, velocities, forces every step.
    pub check_finite: bool,
    /// Clamp penetration depth to this maximum (prevents explosion from deep overlap).
    pub max_penetration: f32,
    /// Maximum allowed mass ratio (heavier/lighter). Pairs exceeding this
    /// get split-island treatment or warning. 0 = disabled.
    pub max_mass_ratio: f32,
    /// Maximum allowed coordinate magnitude. Bodies exceeding this are clamped.
    pub max_coordinate: f32,
    /// Maximum allowed velocity magnitude.
    pub max_velocity: f32,
    /// Maximum allowed angular velocity (rad/s).
    pub max_angular_velocity: f32,
    /// If true, log warnings for detected issues (requires `log` feature).
    pub log_warnings: bool,
    /// If true, automatically correct detected issues (clamp, zero, etc.).
    /// If false, only detect and report (for debugging).
    pub auto_correct: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_finite: cfg!(debug_assertions),
            max_penetration: 1.0,
            max_mass_ratio: 1000.0,
            max_coordinate: 1e4,
            max_velocity: 1e3,
            max_angular_velocity: 1e3,
            log_warnings: false,
            auto_correct: true,
        }
    }
}

impl ValidationConfig {
    /// Strict config for debugging — checks everything, logs everything.
    pub fn strict() -> Self {
        Self {
            check_finite: true,
            max_penetration: 0.5,
            max_mass_ratio: 100.0,
            max_coordinate: 1e3,
            max_velocity: 1e2,
            max_angular_velocity: 1e2,
            log_warnings: true,
            auto_correct: true,
        }
    }

    /// Permissive config for production — minimal overhead, only catastrophic checks.
    pub fn permissive() -> Self {
        Self {
            check_finite: false,
            max_penetration: 10.0,
            max_mass_ratio: 0.0,
            max_coordinate: 1e5,
            max_velocity: 1e4,
            max_angular_velocity: 1e4,
            log_warnings: false,
            auto_correct: true,
        }
    }
}

/// Validation result with optional correction info.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationResult {
    /// No issues detected.
    Ok,
    /// Non-finite value detected and corrected.
    NonFiniteCorrected,
    /// Penetration depth exceeded max and was clamped.
    PenetrationClamped,
    /// Mass ratio exceeded threshold (logged, not corrected).
    MassRatioExceeded,
    /// Coordinate magnitude exceeded max and was clamped.
    CoordinateClamped,
    /// Velocity magnitude exceeded max and was clamped.
    VelocityClamped,
    /// Angular velocity exceeded max and was clamped.
    AngularVelocityClamped,
}

/// Validate and optionally correct a single body's state.
///
/// When `config.log_warnings` is set, each correction is also emitted
/// through the `log` facade (no-op unless the host links a logger).
pub fn validate_body_2d(
    body: &mut crate::body::RigidBody2,
    config: &ValidationConfig,
) -> ValidationResult {
    let mut result = ValidationResult::Ok;

    if config.check_finite {
        let mut has_nonfinite = false;
        if !body.transform.position.x.is_finite() {
            body.transform.position.x = 0.0;
            has_nonfinite = true;
        }
        if !body.transform.position.y.is_finite() {
            body.transform.position.y = 0.0;
            has_nonfinite = true;
        }
        if !body.velocity.x.is_finite() {
            body.velocity.x = 0.0;
            has_nonfinite = true;
        }
        if !body.velocity.y.is_finite() {
            body.velocity.y = 0.0;
            has_nonfinite = true;
        }
        if !body.angular_velocity.is_finite() {
            body.angular_velocity = 0.0;
            has_nonfinite = true;
        }
        if !body.gravity_scale.is_finite() {
            body.gravity_scale = 1.0;
            has_nonfinite = true;
        }
        if has_nonfinite {
            if config.log_warnings {
                log::warn!("body {} had non-finite state; zeroed", body.id);
            }
            result = ValidationResult::NonFiniteCorrected;
        }
    }

    // Clamp position
    let pos_len_sq = body.transform.position.length_squared();
    if pos_len_sq > config.max_coordinate * config.max_coordinate {
        let scale = config.max_coordinate / pos_len_sq.sqrt();
        body.transform.position *= scale;
        if config.log_warnings {
            log::warn!("body {} position exceeded max_coordinate; clamped", body.id);
        }
        result = ValidationResult::CoordinateClamped;
    }

    // Clamp linear velocity
    let vel_len_sq = body.velocity.length_squared();
    if vel_len_sq > config.max_velocity * config.max_velocity {
        let scale = config.max_velocity / vel_len_sq.sqrt();
        body.velocity *= scale;
        if config.log_warnings {
            log::warn!("body {} velocity exceeded max_velocity; clamped", body.id);
        }
        result = ValidationResult::VelocityClamped;
    }

    // Clamp angular velocity
    if body.angular_velocity.abs() > config.max_angular_velocity {
        body.angular_velocity = body.angular_velocity.signum() * config.max_angular_velocity;
        if config.log_warnings {
            log::warn!("body {} angular velocity exceeded max; clamped", body.id);
        }
        result = ValidationResult::AngularVelocityClamped;
    }

    result
}

/// Validate and optionally correct a 3D body's state.
pub fn validate_body_3d(
    body: &mut crate::body::RigidBody3,
    config: &ValidationConfig,
) -> ValidationResult {
    let mut result = ValidationResult::Ok;

    if config.check_finite {
        let mut has_nonfinite = false;
        if !body.transform.position.x.is_finite() {
            body.transform.position.x = 0.0;
            has_nonfinite = true;
        }
        if !body.transform.position.y.is_finite() {
            body.transform.position.y = 0.0;
            has_nonfinite = true;
        }
        if !body.transform.position.z.is_finite() {
            body.transform.position.z = 0.0;
            has_nonfinite = true;
        }
        if !body.velocity.x.is_finite() {
            body.velocity.x = 0.0;
            has_nonfinite = true;
        }
        if !body.velocity.y.is_finite() {
            body.velocity.y = 0.0;
            has_nonfinite = true;
        }
        if !body.velocity.z.is_finite() {
            body.velocity.z = 0.0;
            has_nonfinite = true;
        }
        if !body.angular_velocity.x.is_finite() {
            body.angular_velocity.x = 0.0;
            has_nonfinite = true;
        }
        if !body.angular_velocity.y.is_finite() {
            body.angular_velocity.y = 0.0;
            has_nonfinite = true;
        }
        if !body.angular_velocity.z.is_finite() {
            body.angular_velocity.z = 0.0;
            has_nonfinite = true;
        }
        if !body.gravity_scale.is_finite() {
            body.gravity_scale = 1.0;
            has_nonfinite = true;
        }
        if has_nonfinite {
            if config.log_warnings {
                log::warn!("body {} had non-finite state; zeroed", body.id);
            }
            result = ValidationResult::NonFiniteCorrected;
        }
    }

    let pos_len_sq = body.transform.position.length_squared();
    if pos_len_sq > config.max_coordinate * config.max_coordinate {
        let scale = config.max_coordinate / pos_len_sq.sqrt();
        body.transform.position *= scale;
        if config.log_warnings {
            log::warn!("body {} position exceeded max_coordinate; clamped", body.id);
        }
        result = ValidationResult::CoordinateClamped;
    }

    let vel_len_sq = body.velocity.length_squared();
    if vel_len_sq > config.max_velocity * config.max_velocity {
        let scale = config.max_velocity / vel_len_sq.sqrt();
        body.velocity *= scale;
        if config.log_warnings {
            log::warn!("body {} velocity exceeded max_velocity; clamped", body.id);
        }
        result = ValidationResult::VelocityClamped;
    }

    let ang_vel_len = body.angular_velocity.length();
    if ang_vel_len > config.max_angular_velocity {
        body.angular_velocity *= config.max_angular_velocity / ang_vel_len;
        if config.log_warnings {
            log::warn!("body {} angular velocity exceeded max; clamped", body.id);
        }
        result = ValidationResult::AngularVelocityClamped;
    }

    result
}

/// Validate contact constraints: clamp penetration, check mass ratios.
pub fn validate_constraints_2d(
    bodies: &[crate::body::RigidBody2],
    constraints: &mut [crate::solver::ContactConstraint],
    config: &ValidationConfig,
) -> ValidationResult {
    let mut result = ValidationResult::Ok;

    for c in constraints.iter_mut() {
        // Clamp penetration
        if c.penetration > config.max_penetration {
            c.penetration = config.max_penetration;
            if config.log_warnings {
                log::warn!(
                    "contact ({}, {}) penetration exceeded max; clamped",
                    c.body_a,
                    c.body_b
                );
            }
            result = ValidationResult::PenetrationClamped;
        }
        if c.penetration < 0.0 {
            c.penetration = 0.0;
            if config.log_warnings {
                log::warn!(
                    "contact ({}, {}) had negative penetration; clamped",
                    c.body_a,
                    c.body_b
                );
            }
            result = ValidationResult::PenetrationClamped;
        }

        // Check mass ratio without divisions: heavy/light mass ratio
        // equals light/heavy inverse-mass ratio, so `hi > R * lo` tests
        // the same condition as `hi / lo > R` with one multiply instead
        // of two divisions. Static-involved pairs (zero inverse mass)
        // are skipped exactly as before (their ratio is infinite, and
        // infinite ratios never report). Report-only: no physics impact.
        if config.max_mass_ratio > 0.0 {
            let (ia, ib) = (bodies[c.body_a].inv_mass, bodies[c.body_b].inv_mass);
            if ia > 0.0 && ib > 0.0 {
                let (hi, lo) = if ia > ib { (ia, ib) } else { (ib, ia) };
                if hi > config.max_mass_ratio * lo {
                    if config.log_warnings {
                        log::warn!(
                            "contact ({}, {}) mass ratio exceeded max",
                            c.body_a,
                            c.body_b
                        );
                    }
                    result = ValidationResult::MassRatioExceeded;
                }
            }
        }
    }

    result
}

/// Validate 3D contact constraints.
pub fn validate_constraints_3d(
    bodies: &[crate::body::RigidBody3],
    constraints: &mut [crate::solver3d::ContactConstraint3D],
    config: &ValidationConfig,
) -> ValidationResult {
    let mut result = ValidationResult::Ok;

    for c in constraints.iter_mut() {
        if c.penetration > config.max_penetration {
            c.penetration = config.max_penetration;
            if config.log_warnings {
                log::warn!(
                    "3D contact ({}, {}) penetration exceeded max; clamped",
                    c.body_a,
                    c.body_b
                );
            }
            result = ValidationResult::PenetrationClamped;
        }
        if c.penetration < 0.0 {
            c.penetration = 0.0;
            if config.log_warnings {
                log::warn!(
                    "3D contact ({}, {}) had negative penetration; clamped",
                    c.body_a,
                    c.body_b
                );
            }
            result = ValidationResult::PenetrationClamped;
        }

        if config.max_mass_ratio > 0.0 {
            let (ia, ib) = (bodies[c.body_a].inv_mass, bodies[c.body_b].inv_mass);
            if ia > 0.0 && ib > 0.0 {
                let (hi, lo) = if ia > ib { (ia, ib) } else { (ib, ia) };
                if hi > config.max_mass_ratio * lo {
                    if config.log_warnings {
                        log::warn!(
                            "3D contact ({}, {}) mass ratio exceeded max",
                            c.body_a,
                            c.body_b
                        );
                    }
                    result = ValidationResult::MassRatioExceeded;
                }
            }
        }
    }

    result
}

/// Aggregate validation results for reporting.
#[derive(Debug, Default, Clone, Copy)]
pub struct ValidationReport {
    pub non_finite_corrected: u32,
    pub penetration_clamped: u32,
    pub mass_ratio_exceeded: u32,
    pub coordinate_clamped: u32,
    pub velocity_clamped: u32,
    pub angular_velocity_clamped: u32,
}

impl ValidationReport {
    pub fn merge(&mut self, result: ValidationResult) {
        match result {
            ValidationResult::Ok => {}
            ValidationResult::NonFiniteCorrected => self.non_finite_corrected += 1,
            ValidationResult::PenetrationClamped => self.penetration_clamped += 1,
            ValidationResult::MassRatioExceeded => self.mass_ratio_exceeded += 1,
            ValidationResult::CoordinateClamped => self.coordinate_clamped += 1,
            ValidationResult::VelocityClamped => self.velocity_clamped += 1,
            ValidationResult::AngularVelocityClamped => self.angular_velocity_clamped += 1,
        }
    }

    pub fn any_issues(&self) -> bool {
        self.non_finite_corrected > 0
            || self.penetration_clamped > 0
            || self.mass_ratio_exceeded > 0
            || self.coordinate_clamped > 0
            || self.velocity_clamped > 0
            || self.angular_velocity_clamped > 0
    }
}
