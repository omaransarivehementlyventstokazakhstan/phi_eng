//! The three-dimensional grid.
//!
//! Points are laid out with x fastest: the index of `(ix, iy, iz)` is
//! `ix + nx * (iy + ny * iz)`. The managed mirror uses the same
//! formula, so an index names the same point on both sides.

/// A uniform three-dimensional grid.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid3D {
    nx: usize,
    ny: usize,
    nz: usize,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    z_min: f32,
    z_max: f32,
}

impl Grid3D {
    /// A grid over the three intervals.
    ///
    /// An axis with fewer than two points is clamped to two: a
    /// one-point axis has no spacing, and refusing to be built is
    /// better than dividing by zero later.
    pub fn new(shape: [usize; 3], mins: [f32; 3], maxs: [f32; 3]) -> Self {
        let [nx, ny, nz] = shape;
        let [x_min, y_min, z_min] = mins;
        let [x_max, y_max, z_max] = maxs;
        Self {
            nx: nx.max(2),
            ny: ny.max(2),
            nz: nz.max(2),
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
        }
    }

    /// The point counts along each axis.
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// The total number of points.
    pub fn n(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// The spacings along each axis.
    pub fn spacings(&self) -> (f32, f32, f32) {
        (
            (self.x_max - self.x_min) / (self.nx - 1) as f32,
            (self.y_max - self.y_min) / (self.ny - 1) as f32,
            (self.z_max - self.z_min) / (self.nz - 1) as f32,
        )
    }

    /// The index of `(ix, iy, iz)`, or `None` outside the grid.
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> Option<usize> {
        if ix >= self.nx || iy >= self.ny || iz >= self.nz {
            return None;
        }
        Some(ix + self.nx * (iy + self.ny * iz))
    }

    /// The coordinates of an index: the inverse of
    /// [`Grid3D::index`].
    pub fn coords(&self, index: usize) -> Option<(usize, usize, usize)> {
        if index >= self.n() {
            return None;
        }
        let ix = index % self.nx;
        let iy = (index / self.nx) % self.ny;
        let iz = index / (self.nx * self.ny);
        Some((ix, iy, iz))
    }

    /// The position of an index.
    ///
    /// An index outside the grid returns the far corner.
    pub fn position(&self, index: usize) -> (f32, f32, f32) {
        let (dx, dy, dz) = self.spacings();
        match self.coords(index) {
            Some((ix, iy, iz)) => (
                self.x_min + ix as f32 * dx,
                self.y_min + iy as f32 * dy,
                self.z_min + iz as f32 * dz,
            ),
            None => (self.x_max, self.y_max, self.z_max),
        }
    }

    /// The number of interior unknowns under the Dirichlet condition:
    /// the ends are held at zero on every axis.
    pub fn dirichlet_unknowns(&self) -> usize {
        (self.nx - 2) * (self.ny - 2) * (self.nz - 2)
    }

    /// The grid point of the `u`-th Dirichlet unknown.
    ///
    /// Unknowns run in the same x-fastest order over the interior
    /// points `1..n-1` of each axis. Returns `None` past the last one.
    pub fn unknown_to_grid(&self, u: usize) -> Option<(usize, usize, usize)> {
        let (mx, my, _mz) = (self.nx - 2, self.ny - 2, self.nz - 2);
        if u >= mx * my * (self.nz - 2) {
            return None;
        }
        let ix = u % mx;
        let iy = (u / mx) % my;
        let iz = u / (mx * my);
        Some((ix + 1, iy + 1, iz + 1))
    }

    /// The unknown index of an interior grid point, or `None` on the
    /// boundary: the inverse of [`Grid3D::unknown_to_grid`].
    pub fn grid_to_unknown(&self, ix: usize, iy: usize, iz: usize) -> Option<usize> {
        if ix == 0 || iy == 0 || iz == 0 {
            return None;
        }
        if ix >= self.nx - 1 || iy >= self.ny - 1 || iz >= self.nz - 1 {
            return None;
        }
        let (mx, my) = (self.nx - 2, self.ny - 2);
        Some((ix - 1) + mx * ((iy - 1) + my * (iz - 1)))
    }
}
