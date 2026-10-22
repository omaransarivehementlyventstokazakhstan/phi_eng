using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// Which Bell state.
/// </summary>
public enum BellPair
{
    /// <summary>(|00&gt; + |11&gt;) / sqrt(2).</summary>
    PhiPlus,
    /// <summary>(|00&gt; - |11&gt;) / sqrt(2).</summary>
    PhiMinus,
    /// <summary>(|01&gt; + |10&gt;) / sqrt(2).</summary>
    PsiPlus,
    /// <summary>(|01&gt; - |10&gt;) / sqrt(2).</summary>
    PsiMinus,
}

/// <summary>
/// Entanglement: the Bell states and the CHSH game.
/// </summary>
/// <remarks>
/// This mirrors the Rust <c>phieng_quantum::entanglement</c> module
/// operation for operation, so the two engines agree bit-for-bit.
/// </remarks>
public static class QuantumEntanglement
{
    /// <summary>The Bell state: maximally entangled, perfectly correlated.</summary>
    public static QuantumState BellState(BellPair which)
    {
        float s = QuantumGates.InvSqrt2;
        (float A, float B, float C, float D) = which switch
        {
            BellPair.PhiPlus => (s, 0.0f, 0.0f, s),
            BellPair.PhiMinus => (s, 0.0f, 0.0f, -s),
            BellPair.PsiPlus => (0.0f, s, s, 0.0f),
            BellPair.PsiMinus => (0.0f, s, -s, 0.0f),
            _ => (s, 0.0f, 0.0f, s),
        };
        return new QuantumState(new[]
        {
            new Complex32(A, 0.0f),
            new Complex32(B, 0.0f),
            new Complex32(C, 0.0f),
            new Complex32(D, 0.0f),
        });
    }

    /// <summary>
    /// Plays the CHSH game for <paramref name="rounds"/> rounds with the
    /// given seed and returns the number of rounds won.
    /// </summary>
    /// <remarks>
    /// The stream is seeded, so the same seed gives the same count on
    /// every platform and in both languages: the count is pinned, not
    /// just bounded.
    /// </remarks>
    public static int ChshWinCount(int rounds, ulong seed)
    {
        var rng = new DeterministicRng(seed);
        int wins = 0;
        for (int i = 0; i < rounds; i++)
        {
            if (ChshRound(rng))
            {
                wins++;
            }
        }
        return wins;
    }

    /// <summary>
    /// One round of the CHSH game, played optimally on a fresh Bell pair.
    /// </summary>
    /// <remarks>
    /// Alice measures qubit 0 in the Z basis (a false) or the X basis (a
    /// true), and Bob measures qubit 1 in the (Z + X) / sqrt(2) basis (b
    /// false) or the (Z - X) / sqrt(2) basis (b true). They win the round
    /// when x ^ y == (a &amp;&amp; b).
    /// </remarks>
    private static bool ChshRound(DeterministicRng rng)
    {
        bool a = rng.NextFloat() < 0.5f;
        bool b = rng.NextFloat() < 0.5f;
        var state = BellState(BellPair.PhiPlus);
        // Alice's measurement basis.
        if (a)
        {
            QuantumGates.ApplySingle(state, QuantumGates.H(), 0);
        }
        // Bob's measurement basis: RY(-/+pi/4) turns the tilted axis
        // into Z.
        float angle = b ? MathF.PI / 4.0f : -MathF.PI / 4.0f;
        QuantumGates.ApplySingle(state, QuantumGates.Ry(angle), 1);
        bool x = QuantumGates.Measure(state, 0, rng) ?? false;
        bool y = QuantumGates.Measure(state, 1, rng) ?? false;
        return (x ^ y) == (a && b);
    }
}
