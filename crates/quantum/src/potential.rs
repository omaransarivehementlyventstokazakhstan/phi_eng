//! The potentials a one-dimensional Hamiltonian can carry.

/// The potential energy along a line.
///
/// The units are the engine's: the mass and the reduced Planck constant
/// are one unless the caller says otherwise, so the energies come out
/// in the same units as the grid's length.
#[derive(Clone, Debug, PartialEq)]
pub enum Potential1D {
    /// No potential at all: the free particle.
    Zero,
    /// A constant everywhere.
    Constant(f32),
    /// The harmonic trap, `½ m ω² x²`.
    ///
    /// The energies of a harmonic oscillator on an infinite line are
    /// `(n + ½) hbar ω`; the trap is centred on the origin.
    Harmonic {
        /// The angular frequency.
        omega: f32,
    },
    /// A square well: a flat bottom of `depth` between the walls.
    Box {
        /// The centre of the well.
        center: f32,
        /// Half the width of the well.
        half_width: f32,
        /// The depth of the well (a positive number is a well).
        depth: f32,
    },
    /// A step: `height` on the right of `position`, zero on the left.
    Step {
        /// Where the step is.
        position: f32,
        /// The height of the step.
        height: f32,
    },
    /// Two wells separated by a barrier: the tunnelling system.
    DoubleWell {
        /// The centre of the pair.
        center: f32,
        /// Half the distance between the two wells.
        separation: f32,
        /// The depth of each well.
        depth: f32,
        /// The height of the barrier between them.
        barrier: f32,
    },
    /// The soft Coulomb potential, `-k / sqrt(x² + a²)`.
    ///
    /// The softening `a` keeps the potential finite at the origin, so
    /// the discretisation never sees a singularity. It is the standard
    /// way to make the hydrogen atom solvable on a grid.
    SoftCoulomb {
        /// The strength of the attraction.
        strength: f32,
        /// The softening length.
        softening: f32,
    },
    /// A potential given point by point.
    ///
    /// The values are taken at the grid's points; a shorter list is
    /// padded with zeros, a longer one is truncated.
    Custom(Vec<f32>),
}

impl Potential1D {
    /// The potential energy at `x`.
    pub fn value_at(&self, x: f32) -> f32 {
        match self {
            Potential1D::Zero => 0.0,
            Potential1D::Constant(v) => *v,
            Potential1D::Harmonic { omega } => 0.5 * omega * omega * x * x,
            Potential1D::Box {
                center,
                half_width,
                depth,
            } => {
                if (x - center).abs() <= *half_width {
                    -*depth
                } else {
                    0.0
                }
            }
            Potential1D::Step { position, height } => {
                if x >= *position {
                    *height
                } else {
                    0.0
                }
            }
            Potential1D::DoubleWell {
                center,
                separation,
                depth,
                barrier,
            } => {
                // Two wells of half-width `separation / 2`, with a
                // barrier of the same width between them.
                let half = separation * 0.5;
                let in_well = (x - (center - separation)).abs() <= half
                    || (x - (center + separation)).abs() <= half;
                if in_well {
                    -*depth
                } else if (x - center).abs() <= half {
                    *barrier
                } else {
                    0.0
                }
            }
            Potential1D::SoftCoulomb {
                strength,
                softening,
            } => -strength / (x * x + softening * softening).sqrt(),
            // The custom potential is sampled, not evaluated: its
            // values are indexed by the grid's points, so there is no
            // way to answer for an arbitrary `x`. Use `sample` for it.
            Potential1D::Custom(_) => 0.0,
        }
    }

    /// The potential sampled on a grid.
    ///
    /// The custom potential is the exception: its values are already
    /// sampled, and they are used as they are.
    pub fn sample(&self, grid: &crate::grid::Grid1D) -> Vec<f32> {
        match self {
            Potential1D::Custom(values) => {
                let mut out = values.clone();
                out.resize(grid.n(), 0.0);
                out.truncate(grid.n());
                out
            }
            other => (0..grid.n())
                .map(|i| other.value_at(grid.x_at(i)))
                .collect(),
        }
    }
}
