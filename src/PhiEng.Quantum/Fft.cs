using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// The fast Fourier transform.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::fft</c> module operation for
/// operation, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// The transform is the textbook iterative Cooley-Tukey: a bit reversal
/// permutation followed by <c>log2(n)</c> butterfly stages. The twiddle
/// factors are built from the deterministic trigonometry, never from the
/// platform libm, and the order of every operation is fixed - so the
/// transform gives the same bits on every platform and in both languages.
/// </para>
/// <para>
/// The twiddle angle is computed in <see cref="double"/> and narrowed
/// once, because the angle of the last stage of a long transform loses
/// its low bits in <see cref="float"/>: at <c>2^-20</c> of a radian the
/// phase is already wrong in the seventh digit, and a thousand stages
/// compound it.
/// </para>
/// <para>
/// The length must be a power of two. A transform of any other length
/// is refused rather than approximated: a radix-3 or radix-5 stage would
/// be a second implementation to keep in step, and the quantum grids are
/// powers of two by construction.
/// </para>
/// </remarks>
public static class Fft
{
    /// <summary>Whether a length is a power of two.</summary>
    /// <remarks>
    /// Zero is not a power of two: the transform of nothing is refused.
    /// </remarks>
    public static bool IsPowerOfTwo(int n) => n != 0 && (n & (n - 1)) == 0;

    /// <summary>The transform, in place.</summary>
    /// <param name="data">The buffer to transform.</param>
    /// <param name="inverse">
    /// The direction: the forward transform has no scaling, the inverse
    /// divides by the length at the end. The two are each other's inverse
    /// up to that scaling, so <c>inverse(forward(x)) == x / n</c>.
    /// </param>
    /// <returns>
    /// False when the length is not a power of two or is zero; the buffer
    /// is then undefined.
    /// </returns>
    public static bool TransformInPlace(Complex32[] data, bool inverse) =>
        TransformInPlace(data, data.Length, inverse);

    /// <summary>
    /// The transform of the first <paramref name="count"/> entries, in place.
    /// </summary>
    /// <remarks>
    /// The prefix form is what the transform into a separate buffer needs:
    /// the output buffer may be longer than the input, and the Rust
    /// transforms only the input's length of it.
    /// </remarks>
    internal static bool TransformInPlace(Complex32[] data, int count, bool inverse)
    {
        int n = count;
        if (!IsPowerOfTwo(n))
        {
            return false;
        }
        if (n == 1)
        {
            return true;
        }

        // The bit-reversal permutation: index i goes to the index whose
        // bits are i's bits reversed.
        int j = 0;
        for (int i = 1; i < n; i++)
        {
            int bit = n >> 1;
            while ((j & bit) != 0)
            {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if (i < j)
            {
                (data[i], data[j]) = (data[j], data[i]);
            }
        }

        // The butterfly stages, from the smallest to the largest.
        double sign = inverse ? 1.0 : -1.0;
        int len = 2;
        while (len <= n)
        {
            // The twiddle of the stage: exp(sign * 2 pi i / len).
            float angle = (float)(sign * 2.0 * System.Math.PI / (double)len);
            var (c, s) = DeterministicTrig.ComputeCosSin(angle);
            var w = new Complex32(c, s);
            int half = len / 2;
            int b = 0;
            while (b < n)
            {
                // The twiddle of this butterfly: w^j, built by multiplication
                // so the table is never materialised.
                var wn = Complex32.One;
                for (int k = 0; k < half; k++)
                {
                    Complex32 u = data[b + k];
                    Complex32 v = data[b + k + half] * wn;
                    data[b + k] = u + v;
                    data[b + k + half] = u - v;
                    wn = wn * w;
                }
                b += len;
            }
            len <<= 1;
        }

        if (inverse)
        {
            // The inverse divides by the length.
            float scale = 1.0f / n;
            for (int i = 0; i < n; i++)
            {
                data[i] = Complex32.Scale(data[i], scale);
            }
        }
        return true;
    }

    /// <summary>The transform, into a separate buffer.</summary>
    /// <param name="data">The buffer to transform.</param>
    /// <param name="inverse">The direction of the transform.</param>
    /// <param name="output">
    /// The destination. It must be at least as long as
    /// <paramref name="data"/>; a shorter buffer is left alone and the
    /// function returns false.
    /// </param>
    /// <returns>Whether the transform succeeded.</returns>
    public static bool Transform(Complex32[] data, bool inverse, Complex32[] output)
    {
        if (output.Length < data.Length)
        {
            return false;
        }
        Array.Copy(data, output, data.Length);
        return TransformInPlace(output, data.Length, inverse);
    }

    /// <summary>
    /// The transform of the first <paramref name="count"/> entries of
    /// <paramref name="data"/>, into a separate buffer.
    /// </summary>
    /// <remarks>
    /// The prefix form is what the split operator needs: its workspace is
    /// sized for the largest state it has seen, and only the current
    /// state's length is transformed.
    /// </remarks>
    internal static bool Transform(Complex32[] data, int count, bool inverse, Complex32[] output)
    {
        if (count > data.Length || output.Length < count)
        {
            return false;
        }
        Array.Copy(data, output, count);
        return TransformInPlace(output, count, inverse);
    }
}
