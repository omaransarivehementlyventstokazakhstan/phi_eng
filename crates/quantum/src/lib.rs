//! Quantum mechanics: the states, the operators and the solvers.
//!
//! # What this crate is
//!
//! A quantum state is a vector of complex amplitudes; an observable is a
//! Hermitian operator acting on it; the dynamics is an exponential of
//! that operator. This crate provides the vocabulary and the solvers:
//!
//! * [`state::QuantumState`] — a normalised vector of amplitudes, with
//!   the Born probabilities, the expectation values, the overlap and the
//!   fidelity;
//! * [`operator::QuantumOperator`] — the trait every observable
//!   implements. It is *matrix-free* on purpose: the Hamiltonian of a
//!   wavefunction on a grid is never materialised, it is applied as a
//!   stencil;
//! * [`matrix::ComplexMatrix`] — the dense operators, for the small
//!   systems (a qubit pair, a gate) where a matrix is the natural
//!   representation;
//! * [`linalg`] — the Hermitian Jacobi eigensolver, the power iteration
//!   and the tridiagonal solves the Schrödinger equation needs.
//!
//! # The guarantee
//!
//! The arithmetic is the engine's: `f32` state, the deterministic
//! helpers for every transcendental, and no platform libm anywhere. The
//! managed C# mirror (`src/PhiEng.Quantum`) performs the same
//! operations in the same order, so the two engines agree to the last
//! bit — the same contract the rigid core holds to.
//!
//! The plan for the whole ladder is in `QUANTUM_PLAN.md` at the
//! repository root.

pub mod circuit;
pub mod density;
pub mod entanglement;
pub mod evolution;
pub mod fft;
pub mod field;
pub mod fock;
pub mod gates;
pub mod gauge;
pub mod grid;
pub mod grid3;
pub mod hamiltonian;
pub mod hamiltonian3;
pub mod lattice;
pub mod linalg;
pub mod madelung;
pub mod manybody;
pub mod matrix;
pub mod operator;
pub mod potential;
pub mod scattering;
pub mod spin;
pub mod state;
pub mod variational;

pub use circuit::*;
pub use density::*;
pub use entanglement::*;
pub use evolution::*;
pub use fft::*;
pub use field::*;
pub use fock::*;
pub use gates::*;
pub use gauge::*;
pub use grid::*;
pub use grid3::*;
pub use hamiltonian::*;
pub use hamiltonian3::*;
pub use lattice::*;
pub use linalg::*;
pub use madelung::*;
pub use manybody::*;
pub use matrix::*;
pub use operator::*;
pub use potential::*;
pub use scattering::*;
pub use spin::*;
pub use state::*;
pub use variational::*;
