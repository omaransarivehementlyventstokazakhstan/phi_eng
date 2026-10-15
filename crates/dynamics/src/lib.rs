pub mod body;
pub mod fluid;
pub mod fluid3;
pub(crate) mod fxhash;
pub mod island;
pub mod joint;
pub mod joint3;
pub mod material;
pub mod pool;
pub mod snapshot;
pub mod solver;
pub mod solver3d;
pub mod validation;
pub mod world;
// NOTE: world_soa is an experimental SoA prototype, currently disabled.
// It duplicates World2 narrowphase/solver logic and does not compile against
// Solver::solve(&mut Vec<RigidBody2>). See Phase B notes. Re-enable only
// when the SoA solver backend is ready with differential tests.
// pub mod world_soa;

pub use body::*;
pub use fluid::*;
pub use fluid3::*;
pub use island::*;
pub use joint::*;
pub use joint3::*;
pub use material::*;
pub use pool::*;
pub use snapshot::*;
pub use solver::*;
pub use solver3d::*;
pub use validation::*;
pub use world::*;
// pub use world_soa::*;
