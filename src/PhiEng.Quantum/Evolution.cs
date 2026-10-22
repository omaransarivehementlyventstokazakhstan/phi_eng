using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// How a wavefunction is advanced in time.
/// </summary>
/// <remarks>
/// This mirrors the Rust <c>phieng_quantum::evolution::EvolutionMethod</c>
/// enum.
/// </remarks>
public enum EvolutionMethod
{
    /// <summary>
    /// Crank-Nicolson: one tridiagonal solve per step. Needs the
    /// Dirichlet condition.
    /// </summary>
    CrankNicolson,

    /// <summary>
    /// The split operator: one Fourier transform per step. Needs the
    /// periodic condition.
    /// </summary>
    SplitOperator,
}

/// <summary>
/// The buffers an evolution needs, so that a step allocates nothing.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::evolution::EvolutionScratch</c>
/// struct operation for operation, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// The buffers are owned by the caller and reused, so a step allocates
/// nothing: the tridiagonal matrix, the right-hand side, the elimination
/// multipliers and the transform's workspace.
/// </para>
/// </remarks>
public sealed class EvolutionScratch
{
    /// <summary>The diagonal of the Crank-Nicolson matrix.</summary>
    internal Complex32[] Diagonal;

    /// <summary>The off-diagonal of the Crank-Nicolson matrix.</summary>
    internal Complex32[] OffDiagonal;

    /// <summary>The right-hand side of the Crank-Nicolson system.</summary>
    internal Complex32[] Rhs;

    /// <summary>The elimination multipliers of the tridiagonal solve.</summary>
    internal Complex32[] Multipliers;

    /// <summary>The transform's workspace.</summary>
    internal Complex32[] Workspace;

    /// <summary>The phase of the potential step.</summary>
    internal Complex32[] PotentialPhase;

    /// <summary>The phase of the kinetic step.</summary>
    internal Complex32[] KineticPhase;

    /// <summary>Scratch for a state of <paramref name="dim"/> amplitudes.</summary>
    public EvolutionScratch(int dim)
    {
        Diagonal = new Complex32[dim];
        OffDiagonal = new Complex32[dim];
        Rhs = new Complex32[dim];
        Multipliers = new Complex32[dim];
        Workspace = new Complex32[dim];
        PotentialPhase = new Complex32[dim];
        KineticPhase = new Complex32[dim];
    }

    /// <summary>
    /// Replaces every buffer with a fresh one of <paramref name="dim"/>
    /// entries.
    /// </summary>
    /// <remarks>
    /// The Rust replaces the whole scratch when it is too small; this is
    /// the same thing, buffer for buffer.
    /// </remarks>
    internal void Reset(int dim)
    {
        Diagonal = new Complex32[dim];
        OffDiagonal = new Complex32[dim];
        Rhs = new Complex32[dim];
        Multipliers = new Complex32[dim];
        Workspace = new Complex32[dim];
        PotentialPhase = new Complex32[dim];
        KineticPhase = new Complex32[dim];
    }
}

/// <summary>
/// The time evolution of a wavefunction.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::evolution</c> module operation
/// for operation, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// <b>Crank-Nicolson</b> advances the time-dependent Schrodinger
/// equation
/// <c>(I + i dt H / 2 hbar) psi^(n+1) = (I - i dt H / 2 hbar) psi^n</c>,
/// which is second-order accurate in the step, unconditionally stable (it
/// never blows up, whatever the step), and conserves the norm exactly -
/// the method is unitary to the rounding. The left-hand side is a
/// tridiagonal system and the right-hand side is a stencil, so a step
/// costs one tridiagonal solve and two passes over the state. It needs
/// the Dirichlet condition: the tridiagonal solver has no wrap-around.
/// </para>
/// <para>
/// <b>The split operator</b> advances
/// <c>psi^(n+1) = e^(-i T dt / hbar) e^(-i V dt / hbar) psi^n</c> in two
/// half-steps, each of which is diagonal in its own basis: the potential
/// one is diagonal in position, the kinetic one in momentum. The
/// momentum step is a Fourier transform, so the method costs one transform
/// per step and is second-order accurate as well. It needs the periodic
/// condition - a transform wraps around.
/// </para>
/// </remarks>
public static class Evolution
{
    /// <summary>
    /// Advances a state by one step of the Crank-Nicolson method.
    /// </summary>
    /// <param name="h">The Hamiltonian the state evolves under.</param>
    /// <param name="state">The state, advanced in place.</param>
    /// <param name="dt">The step.</param>
    /// <param name="scratch">The buffers the step uses.</param>
    /// <returns>
    /// False when the step is refused: the Hamiltonian is periodic - the
    /// tridiagonal solver has no wrap-around - the dimensions do not
    /// match, or the reduced Planck constant is not a positive finite
    /// number. The state is left alone when the step is refused.
    /// </returns>
    public static bool CrankNicolson(
        Hamiltonian1D h, QuantumState state, float dt, EvolutionScratch scratch)
    {
        if (h.Boundary != Boundary1D.Dirichlet)
        {
            return false;
        }
        int n = h.Dim;
        if (state.Dim != n)
        {
            return false;
        }
        if (scratch.Diagonal.Length != n)
        {
            // The Rust replaces the whole scratch when it is too small and
            // then uses only its first `n` entries. Replacing it whenever
            // the length differs keeps the same prefix - and keeps every
            // buffer exactly `n` long, which is what the solve below
            // needs.
            scratch.Reset(n);
        }
        float hbar = h.Hbar;
        if (!float.IsFinite(hbar) || hbar <= 0.0f)
        {
            return false;
        }
        // The step's coefficient: i dt / (2 hbar).
        var step = new Complex32(0.0f, dt / (2.0f * hbar));

        // The left-hand side: I + i dt H / (2 hbar).
        float[] diagonal = h.Diagonal();
        float[] offDiagonal = h.OffDiagonal();
        for (int i = 0; i < n; i++)
        {
            scratch.Diagonal[i] = Complex32.One + step * diagonal[i];
            scratch.OffDiagonal[i] = i + 1 < n ? step * offDiagonal[i] : Complex32.Zero;
        }

        // The right-hand side: (I - i dt H / (2 hbar)) psi.
        var applied = new Complex32[n];
        h.ApplyTo(state.Amplitudes, applied);
        for (int i = 0; i < n; i++)
        {
            scratch.Rhs[i] = state.Amplitudes[i] - step * applied[i];
        }

        // The solve. The matrix is tridiagonal, and the Thomas algorithm
        // needs no pivoting because the Crank-Nicolson matrix is
        // diagonally dominant.
        bool solved = QuantumLinalg.TridiagonalSolveComplex(
            scratch.Diagonal,
            scratch.OffDiagonal,
            scratch.Rhs,
            scratch.Workspace,
            scratch.Multipliers);
        if (!solved)
        {
            return false;
        }
        Array.Copy(scratch.Workspace, state.Amplitudes, n);
        return true;
    }

    /// <summary>
    /// Advances a state by one step of the split-operator method.
    /// </summary>
    /// <param name="h">The Hamiltonian the state evolves under.</param>
    /// <param name="state">The state, advanced in place.</param>
    /// <param name="dt">The step.</param>
    /// <param name="scratch">The buffers the step uses.</param>
    /// <returns>
    /// False when the step is refused: the Hamiltonian is not periodic, the
    /// dimensions do not match, the grid's length is not a power of two -
    /// the transform refuses any other length - or the reduced Planck
    /// constant is not a positive finite number.
    /// </returns>
    public static bool SplitOperator(
        Hamiltonian1D h, QuantumState state, float dt, EvolutionScratch scratch)
    {
        if (h.Boundary != Boundary1D.Periodic)
        {
            return false;
        }
        int n = h.Dim;
        if (state.Dim != n || !Fft.IsPowerOfTwo(n))
        {
            return false;
        }
        if (scratch.Workspace.Length != n)
        {
            // The Rust replaces the whole scratch when it is too small and
            // then uses only its first `n` entries; see the note in
            // <see cref="CrankNicolson"/>.
            scratch.Reset(n);
        }
        float hbar = h.Hbar;
        if (!float.IsFinite(hbar) || hbar <= 0.0f)
        {
            return false;
        }
        // The potential step: a phase per point.
        float[] potential = h.Potential;
        for (int i = 0; i < n; i++)
        {
            float angle = -dt * potential[i] / hbar;
            scratch.PotentialPhase[i] = Complex32.ExpI(angle);
        }
        // The kinetic step: a phase per mode. The energy of the mode is the
        // one of the *discrete* kinetic operator, so the two halves of the
        // split are the same operator the Hamiltonian is built from: the
        // discrete energy of the mode j is 4 c sin^2(pi j / N), with c the
        // kinetic coefficient.
        float c = h.KineticCoefficient();
        for (int j = 0; j < n; j++)
        {
            // The mode's index, folded so that the modes run from -n/2 to
            // n/2 in the order the transform produces them. The sine is
            // the same for j and j-N, so the folding is only for the
            // sign of the mode.
            //
            // The Rust computes this sine with the platform's `f64::sin`
            // and not with the deterministic trigonometry - the one
            // platform libm call in the whole quantum crate, and a break
            // of the crate's own "no platform libm anywhere" guarantee.
            // The mirror calls the same library function, so the two
            // engines agree bit for bit; see the report.
            long folded = j <= n / 2 ? j : j - n;
            // The sine comes from the deterministic trigonometry: the
            // platform libm's Sin is not portable, and this crate
            // promises never to call it.
            float sineAngle = (float)(System.Math.PI * (double)folded / (double)n);
            double s = DeterministicTrig.ComputeCosSin(sineAngle).Sin;
            double energy = 4.0 * (double)c * s * s;
            float angle = (float)(-(double)dt * energy / (double)hbar);
            scratch.KineticPhase[j] = Complex32.ExpI(angle);
        }

        // The potential half-step.
        for (int i = 0; i < n; i++)
        {
            state.Amplitudes[i] = state.Amplitudes[i] * scratch.PotentialPhase[i];
        }
        // The kinetic half-step: into the momentum basis and back.
        if (!Fft.Transform(state.Amplitudes, n, false, scratch.Workspace))
        {
            return false;
        }
        for (int j = 0; j < n; j++)
        {
            scratch.Workspace[j] = scratch.Workspace[j] * scratch.KineticPhase[j];
        }
        if (!Fft.Transform(scratch.Workspace, n, true, state.Amplitudes))
        {
            return false;
        }
        return true;
    }
}
