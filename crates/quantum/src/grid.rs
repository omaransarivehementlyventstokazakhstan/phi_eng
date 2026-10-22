//! The one-dimensional grid.

/// A uniform grid of `n` points between `x_min` and `x_max`.
///
/// The spacing is `(x_max - x_min) / (n - 1)`: the two ends are
/// *points*, not the edges of a cell, so the grid includes both
/// boundary values.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid1D {
    n: usize,
    x_min: f32,
    x_max: f32,
}

impl Grid1D {
    /// A grid of `n` points from `x_min` to `x_max`.
    ///
    /// A grid of fewer than two points has no spacing and is clamped to
    /// two: a one-point grid is a degenerate case the solvers cannot
    /// use, and refusing to be built is better than dividing by zero
    /// later.
    pub fn new(n: usize, x_min: f32, x_max: f32) -> Self {
        Self {
            n: n.max(2),
            x_min,
            x_max,
        }
    }

    /// The number of points.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The left end.
    pub fn x_min(&self) -> f32 {
        self.x_min
    }

    /// The right end.
    pub fn x_max(&self) -> f32 {
        self.x_max
    }

    /// The spacing between the points.
    pub fn dx(&self) -> f32 {
        (self.x_max - self.x_min) / (self.n - 1) as f32
    }

    /// The position of the `index`-th point.
    ///
    /// An index outside the grid returns the nearest end.
    pub fn x_at(&self, index: usize) -> f32 {
        if index >= self.n {
            return self.x_max;
        }
        self.x_min + index as f32 * self.dx()
    }

    /// Every point's position.
    pub fn positions(&self) -> Vec<f32> {
        (0..self.n).map(|i| self.x_at(i)).collect()
    }
}
