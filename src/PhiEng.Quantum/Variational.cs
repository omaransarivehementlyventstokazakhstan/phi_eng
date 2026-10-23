namespace PhiEng.Quantum;

/// <summary>
/// Variational loops: parameter-shift gradients and the descent.
/// </summary>
/// <remarks>
/// This mirrors the Rust <c>phieng_quantum::variational</c> module
/// argument for argument, so the two engines agree bit-for-bit.
/// </remarks>
public static class QuantumVariational
{
    /// <summary>The exact gradient of <paramref name="expectation"/> at <paramref name="theta"/>, by the parameter shift.</summary>
    /// <remarks>
    /// Half the difference between the expectations at θ+π/2 and
    /// θ−π/2: exact for Pauli rotations, because sine and cosine are
    /// their own shifted differences.
    /// </remarks>
    public static float ParameterShiftGradient(System.Func<float, float> expectation, float theta)
    {
        float halfPi = MathF.PI / 2.0f;
        return 0.5f * (expectation(theta + halfPi) - expectation(theta - halfPi));
    }

    /// <summary>Gradient descent with a fixed step count and rate: deterministic to the bit.</summary>
    /// <returns>The final parameters beside the final energy.</returns>
    public static (float[] Params, float Energy) Descend(
        float[] initial, int steps, float rate,
        System.Func<float[], float> energy,
        System.Func<float[], float[]> gradient)
    {
        var @params = (float[])initial.Clone();
        for (int s = 0; s < steps; s++)
        {
            float[] g = gradient(@params);
            for (int i = 0; i < @params.Length; i++)
            {
                @params[i] -= rate * g[i];
            }
        }
        return (@params, energy(@params));
    }
}
