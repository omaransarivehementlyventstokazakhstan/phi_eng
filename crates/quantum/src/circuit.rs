//! The quantum circuit: a register, a seeded measurement stream and the
//! algorithms.
//!
//! A circuit owns an `n`-qubit register and the random stream its
//! measurements draw from, so a whole algorithm — prepare, evolve,
//! measure — runs inside one value with one seed. Running it twice with
//! the same seed gives the same shots.

use phieng_math::complex::Complex32;
use phieng_math::rng::DeterministicRng;

use crate::gates;
use crate::matrix::ComplexMatrix;
use crate::state::QuantumState;

/// A truth table for a Boolean oracle on `n` inputs.
///
/// The table lists `f(x)` for every `x` in ascending order. An oracle
/// is data, not code, so the two engines cannot disagree on what it
/// computes: the same table in, the same unitary out.
#[derive(Clone, Debug, PartialEq)]
pub struct TruthTable {
    inputs: usize,
    table: Vec<bool>,
}

impl TruthTable {
    /// A table of `2^n_inputs` constant values.
    pub fn constant(n_inputs: usize, value: bool) -> Self {
        Self {
            inputs: n_inputs,
            table: vec![value; 1usize << n_inputs.min(20)],
        }
    }

    /// The balanced oracle `f(x) = x_0`: the value is the least
    /// significant input bit.
    pub fn first_bit(n_inputs: usize) -> Self {
        let mut table = Vec::with_capacity(1usize << n_inputs.min(20));
        for x in 0..(1usize << n_inputs.min(20)) {
            table.push(x & 1 != 0);
        }
        Self {
            inputs: n_inputs,
            table,
        }
    }

    /// The Bernstein-Vazirani oracle `f(x) = parity(secret & x)`.
    pub fn dot_product(n_inputs: usize, secret: u32) -> Self {
        let mut table = Vec::with_capacity(1usize << n_inputs.min(20));
        for x in 0..(1usize << n_inputs.min(20)) {
            table.push(((x as u32 & secret).count_ones() & 1) != 0);
        }
        Self {
            inputs: n_inputs,
            table,
        }
    }

    /// The number of input bits.
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// The oracle's value on `x`.
    pub fn value(&self, x: usize) -> bool {
        self.table.get(x).copied().unwrap_or(false)
    }
}

/// A register of `n` qubits with a seeded measurement stream.
#[derive(Clone, Debug)]
pub struct QuantumCircuit {
    qubits: usize,
    state: QuantumState,
    rng: DeterministicRng,
}

impl QuantumCircuit {
    /// A circuit of `n` qubits in `|0...0>`, drawing from `seed`.
    pub fn new(qubits: usize, seed: u64) -> Self {
        let mut state = QuantumState::zeros(1usize << qubits.min(20));
        state.amplitudes_mut()[0] = Complex32::one();
        Self {
            qubits,
            state,
            rng: DeterministicRng::new(seed),
        }
    }

    /// The number of qubits.
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    /// The register's state.
    pub fn state(&self) -> &QuantumState {
        &self.state
    }

    /// The register's state, mutable: a caller may prepare a state of
    /// its own in the circuit.
    pub fn state_mut(&mut self) -> &mut QuantumState {
        &mut self.state
    }

    /// Applies a 2 by 2 gate to one qubit.
    pub fn apply_single(&mut self, gate: &ComplexMatrix, qubit: usize) -> bool {
        gates::apply_single(&mut self.state, gate, qubit)
    }

    /// Applies a CNOT.
    pub fn apply_cnot(&mut self, control: usize, target: usize) -> bool {
        gates::apply_cnot(&mut self.state, control, target)
    }

    /// Applies a Toffoli.
    pub fn apply_toffoli(&mut self, a: usize, b: usize, target: usize) -> bool {
        gates::apply_toffoli(&mut self.state, a, b, target)
    }

    /// Applies a SWAP.
    pub fn apply_swap(&mut self, a: usize, b: usize) -> bool {
        gates::apply_swap(&mut self.state, a, b)
    }

    /// Applies a controlled phase.
    pub fn apply_controlled_phase(&mut self, control: usize, target: usize, theta: f32) -> bool {
        gates::apply_controlled_phase(&mut self.state, control, target, theta)
    }

    /// Applies a bit-flip oracle: `|x>|y>` becomes `|x>|y ^ f(x)>`,
    /// with the ancilla the highest qubit.
    ///
    /// The table must hold `2^n` entries for the `n` lowest qubits.
    /// Tables above twenty inputs are refused: the register itself is
    /// capped there.
    pub fn apply_oracle(&mut self, oracle: &TruthTable) -> bool {
        let n = oracle.inputs();
        if n > 20 || self.qubits < n + 1 {
            return false;
        }
        let ancilla = 1usize << n;
        let dim = self.state.dim();
        let amps = self.state.amplitudes_mut();
        for x in 0..(1usize << n) {
            if !oracle.value(x) {
                continue;
            }
            let i = x;
            let j = x | ancilla;
            if j >= dim {
                return false;
            }
            amps.swap(i, j);
        }
        true
    }

    /// Measures one qubit, collapsing the register.
    pub fn measure(&mut self, qubit: usize) -> Option<bool> {
        gates::measure(&mut self.state, qubit, &mut self.rng)
    }

    /// Measures every qubit, lowest first.
    pub fn measure_all(&mut self) -> Vec<bool> {
        let mut out = Vec::with_capacity(self.qubits);
        for q in 0..self.qubits {
            out.push(self.measure(q).unwrap_or(false));
        }
        out
    }

    /// The probability of each basis state.
    pub fn probabilities(&self) -> Vec<f32> {
        self.state.born_probabilities()
    }
}

/// The Deutsch-Jozsa problem: is the oracle constant or balanced?
///
/// Returns true for a constant oracle. The algorithm uses one query;
/// a classical algorithm needs exponentially many in the worst case.
pub fn deutsch_jozsa(n_inputs: usize, oracle: &TruthTable, seed: u64) -> bool {
    let mut circuit = QuantumCircuit::new(n_inputs + 1, seed);
    let ancilla = n_inputs;
    // The ancilla starts in |1>.
    circuit.state.amplitudes_mut()[0] = Complex32::zero();
    circuit.state.amplitudes_mut()[1usize << ancilla] = Complex32::one();
    // Hadamards everywhere.
    let h = gates::h_gate();
    for q in 0..=n_inputs {
        circuit.apply_single(&h, q);
    }
    circuit.apply_oracle(oracle);
    for q in 0..n_inputs {
        circuit.apply_single(&h, q);
    }
    // Constant oracles leave the inputs in |0...0>.
    let outcomes = circuit.measure_all();
    outcomes[..n_inputs].iter().all(|b| !b)
}

/// The Bernstein-Vazirani problem: recover the secret string with one
/// query.
pub fn bernstein_vazirani(secret: u32, n_inputs: usize, seed: u64) -> u32 {
    let oracle = TruthTable::dot_product(n_inputs, secret);
    let mut circuit = QuantumCircuit::new(n_inputs + 1, seed);
    let ancilla = n_inputs;
    circuit.state.amplitudes_mut()[0] = Complex32::zero();
    circuit.state.amplitudes_mut()[1usize << ancilla] = Complex32::one();
    let h = gates::h_gate();
    for q in 0..=n_inputs {
        circuit.apply_single(&h, q);
    }
    circuit.apply_oracle(&oracle);
    for q in 0..n_inputs {
        circuit.apply_single(&h, q);
    }
    let mut recovered = 0u32;
    for q in 0..n_inputs {
        if circuit.measure(q).unwrap_or(false) {
            recovered |= 1 << q;
        }
    }
    recovered
}

/// The number of Grover iterations for `n_qubits`: `floor(pi/4 sqrt(N))`.
///
/// The floor (not the ceiling) is what lands on the target: for four
/// items one iteration rotates the state exactly onto the marked item,
/// and the ceiling would overshoot it back down.
pub fn grover_iterations(n_qubits: usize) -> usize {
    let n = (1usize << n_qubits.min(20)) as f64;
    ((std::f64::consts::PI / 4.0) * n.sqrt()) as usize
}

/// The Grover search: the probability of the marked item after the
/// prescribed number of iterations.
///
/// The oracle flips the target's phase; the diffusion reflects about
/// the mean. Both are applied by index manipulation, in ascending
/// order, so the two engines agree bit-for-bit.
pub fn grover_probability(n_qubits: usize, target: usize) -> f32 {
    let dim = 1usize << n_qubits.min(20);
    if target >= dim {
        return 0.0;
    }
    let mut circuit = QuantumCircuit::new(n_qubits, 1);
    let h = gates::h_gate();
    for q in 0..n_qubits {
        circuit.apply_single(&h, q);
    }
    let iterations = grover_iterations(n_qubits).max(1);
    for _ in 0..iterations {
        // The phase oracle: the target changes sign.
        let target_amp = circuit.state.amplitudes()[target];
        circuit.state.amplitudes_mut()[target] = -target_amp;
        // The diffusion: reflect about the mean.
        let mut mean = Complex32::zero();
        for a in circuit.state.amplitudes().iter() {
            mean = mean + *a;
        }
        mean = mean.scale(1.0 / dim as f32);
        for a in circuit.state.amplitudes_mut().iter_mut() {
            *a = mean.scale(2.0) - *a;
        }
    }
    circuit.state.amplitudes()[target].norm_sqr()
}

/// The quantum Fourier transform, in place.
///
/// The circuit is Hadamards interleaved with controlled rotations and
/// closed with the swaps that undo the bit reversal, so applying it
/// and then its inverse returns the input.
pub fn apply_qft(circuit: &mut QuantumCircuit, inverse: bool) {
    let n = circuit.qubits();
    let h = gates::h_gate();
    if !inverse {
        for j in (0..n).rev() {
            circuit.apply_single(&h, j);
            for k in (0..j).rev() {
                let power = (j - k + 1) as i32;
                let angle = 2.0 * std::f32::consts::PI / (1u32 << power.min(30)) as f32;
                circuit.apply_controlled_phase(k, j, angle);
            }
        }
        for j in 0..n / 2 {
            circuit.apply_swap(j, n - 1 - j);
        }
    } else {
        // The adjoint in reverse order: F = S U, so F† = U† S, and the
        // swaps -- their own inverse -- come first. Controlled phases
        // on the same target commute, so each target undoes its
        // rotations and then its Hadamard.
        for j in 0..n / 2 {
            circuit.apply_swap(j, n - 1 - j);
        }
        for j in 0..n {
            for k in 0..j {
                let power = (j - k + 1) as i32;
                let angle = -2.0 * std::f32::consts::PI / (1u32 << power.min(30)) as f32;
                circuit.apply_controlled_phase(k, j, angle);
            }
            circuit.apply_single(&h, j);
        }
    }
}

/// The teleportation protocol: the fidelity of Bob's qubit with the
/// input state after Alice's measurements and the corrections.
///
/// The fidelity is one whatever Alice measures, so the seeded stream
/// only decides which corrections run, not the answer.
pub fn teleport_fidelity(input: &QuantumState, seed: u64) -> f32 {
    if input.dim() != 2 {
        return 0.0;
    }
    // Qubit 0 carries the input, qubits 1 and 2 share a Bell pair.
    let mut circuit = QuantumCircuit::new(3, seed);
    for i in 0..2 {
        circuit.state.amplitudes_mut()[i] = input.amplitudes()[i];
    }
    let h = gates::h_gate();
    circuit.apply_single(&h, 1);
    circuit.apply_cnot(1, 2);
    // Alice's Bell measurement.
    circuit.apply_cnot(0, 1);
    circuit.apply_single(&h, 0);
    let m0 = circuit.measure(0).unwrap_or(false);
    let m1 = circuit.measure(1).unwrap_or(false);
    // The corrections on Bob's qubit.
    if m1 {
        let x = gates::x_gate();
        circuit.apply_single(&x, 2);
    }
    if m0 {
        let z = gates::z_gate();
        circuit.apply_single(&z, 2);
    }
    // Bob's qubit, read off the collapsed state: Alice's two qubits
    // are fixed, so the amplitudes at the two indices that differ only
    // in qubit 2 are his state up to the known branch.
    let base = (m0 as usize) | ((m1 as usize) << 1);
    let b0 = circuit.state.amplitudes()[base];
    let b1 = circuit.state.amplitudes()[base | 4];
    let mut bob = QuantumState::from_amplitudes(vec![b0, b1]);
    bob.normalize();
    bob.fidelity(input)
}

/// The superdense coding protocol: two classical bits travel on one
/// qubit plus a shared Bell pair.
///
/// Returns the two bits Bob decodes. The final state is a basis state,
/// so the seeded stream never matters — but it is still seeded, so the
/// run is reproducible bit-for-bit.
pub fn superdense_send(bit0: bool, bit1: bool) -> (bool, bool) {
    let mut circuit = QuantumCircuit::new(2, 1);
    let h = gates::h_gate();
    circuit.apply_single(&h, 0);
    circuit.apply_cnot(0, 1);
    // Alice encodes: X for the first bit, Z for the second.
    if bit0 {
        let x = gates::x_gate();
        circuit.apply_single(&x, 0);
    }
    if bit1 {
        let z = gates::z_gate();
        circuit.apply_single(&z, 0);
    }
    // Bob decodes: CNOT, Hadamard, measure. The first measured bit
    // carries the Z-encoded bit and the second the X-encoded one, so
    // the pair comes back swapped relative to the encoding order.
    circuit.apply_cnot(0, 1);
    circuit.apply_single(&h, 0);
    let m0 = circuit.measure(0).unwrap_or(false);
    let m1 = circuit.measure(1).unwrap_or(false);
    (m1, m0)
}
