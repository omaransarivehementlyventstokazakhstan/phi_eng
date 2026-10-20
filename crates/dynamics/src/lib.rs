pub mod ambient;
pub mod barnes_hut;
pub mod body;
pub mod ccd;
pub mod collision_tier;
pub mod fluid;
pub mod fluid3;
pub(crate) mod fxhash;
pub mod gravity;
pub mod island;
pub mod joint;
pub mod joint3;
pub mod material;
pub mod noise;
pub mod physics_config;
pub mod pool;
pub mod radiation;
pub mod snapshot;
pub mod solver;
pub mod solver3d;
pub mod thermal;
pub mod validation;
pub mod world;
// NOTE: world_soa is an experimental SoA prototype, currently disabled.
// It duplicates World2 narrowphase/solver logic and does not compile against
// Solver::solve(&mut Vec<RigidBody2>). See Phase B notes. Re-enable only
// when the SoA solver backend is ready with differential tests.
// pub mod world_soa;

pub use ambient::*;
pub use barnes_hut::*;
pub use body::*;
pub use ccd::*;
pub use collision_tier::*;
pub use fluid::*;
pub use fluid3::*;
pub use gravity::*;
pub use island::*;
pub use joint::*;
pub use joint3::*;
pub use material::*;
pub use noise::*;
pub use physics_config::*;
pub use pool::*;
pub use snapshot::*;
pub use solver::*;
pub use solver3d::*;
pub use thermal::*;
pub use validation::*;
pub use world::*;
// pub use world_soa::*;
