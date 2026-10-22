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

pub mod grid;
pub mod hamiltonian;
pub mod linalg;
pub mod matrix;
pub mod operator;
pub mod potential;
pub mod state;

pub use grid::*;
pub use hamiltonian::*;
pub use linalg::*;
pub use matrix::*;
pub use operator::*;
pub use potential::*;
pub use state::*;
