using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>
/// Deterministic value noise for ambient turbulence. Mirrors Rust
/// <c>phieng_dynamics::noise</c>. Every entry point is a pure function of
/// (position, time, seed): no RNG, no wall clock. Cross-language
/// bit-equality with Rust is NOT claimed (different float ops) — parity
/// tests bind to statistical bounds.
/// </summary>
public static class TurbulenceNoise
{
    private const uint TimePrime = 0x9e3779b9u;

    private static uint HashU32(uint x)
    {
        unchecked
        {
            x ^= x >> 16;
            x *= 0x7feb352du;
            x ^= x >> 15;
            x *= 0x846ca68bu;
            x ^= x >> 16;
            return x;
        }
    }

    private static float Lattice2(int ix, int iy, uint seed)
    {
        unchecked
        {
            uint h = HashU32((uint)ix * 0x1f1f1f1fu + (uint)iy * 0x9e3779b9u + seed);
            return (h >> 8) * (1.0f / 16777216.0f);
        }
    }

    private static float Lattice3(int ix, int iy, int iz, uint seed)
    {
        unchecked
        {
            uint h = HashU32(
                (uint)ix * 0x1f1f1f1fu + (uint)iy * 0x9e3779b9u + (uint)iz * 0x85ebca6bu + seed);
            return (h >> 8) * (1.0f / 16777216.0f);
        }
    }

    private static float Fade(float t) => t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);

    private static float Lerp(float a, float b, float t) => a + (b - a) * t;

    /// <summary>2D value noise in [0, 1).</summary>
    public static float ValueNoise2(float x, float y, uint seed)
    {
        float x0 = MathF.Floor(x);
        float y0 = MathF.Floor(y);
        int xi = (int)x0;
        int yi = (int)y0;
        float tx = Fade(x - x0);
        float ty = Fade(y - y0);
        float v00 = Lattice2(xi, yi, seed);
        float v10 = Lattice2(xi + 1, yi, seed);
        float v01 = Lattice2(xi, yi + 1, seed);
        float v11 = Lattice2(xi + 1, yi + 1, seed);
        return Lerp(Lerp(v00, v10, tx), Lerp(v01, v11, tx), ty);
    }

    /// <summary>3D value noise in [0, 1).</summary>
    public static float ValueNoise3(float x, float y, float z, uint seed)
    {
        float x0 = MathF.Floor(x);
        float y0 = MathF.Floor(y);
        float z0 = MathF.Floor(z);
        int xi = (int)x0;
        int yi = (int)y0;
        int zi = (int)z0;
        float tx = Fade(x - x0);
        float ty = Fade(y - y0);
        float tz = Fade(z - z0);
        float v000 = Lattice3(xi, yi, zi, seed);
        float v100 = Lattice3(xi + 1, yi, zi, seed);
        float v010 = Lattice3(xi, yi + 1, zi, seed);
        float v110 = Lattice3(xi + 1, yi + 1, zi, seed);
        float v001 = Lattice3(xi, yi, zi + 1, seed);
        float v101 = Lattice3(xi + 1, yi, zi + 1, seed);
        float v011 = Lattice3(xi, yi + 1, zi + 1, seed);
        float v111 = Lattice3(xi + 1, yi + 1, zi + 1, seed);
        float a = Lerp(Lerp(v000, v100, tx), Lerp(v010, v110, tx), ty);
        float b = Lerp(Lerp(v001, v101, tx), Lerp(v011, v111, tx), ty);
        return Lerp(a, b, tz);
    }

    private static float Fbm2Static(float x, float y, uint seed, int octaves)
    {
        float sum = 0.0f;
        float amp = 1.0f;
        float freq = 1.0f;
        float norm = 0.0f;
        for (int o = 0; o < octaves; o++)
        {
            sum += amp * (ValueNoise2(x * freq, y * freq, seed + (uint)o) * 2.0f - 1.0f);
            norm += amp;
            amp *= 0.5f;
            freq *= 2.0f;
        }
        return norm > 0.0f ? sum / norm : 0.0f;
    }

    private static float Fbm3Static(float x, float y, float z, uint seed, int octaves)
    {
        float sum = 0.0f;
        float amp = 1.0f;
        float freq = 1.0f;
        float norm = 0.0f;
        for (int o = 0; o < octaves; o++)
        {
            sum += amp * (ValueNoise3(x * freq, y * freq, z * freq, seed + (uint)o) * 2.0f - 1.0f);
            norm += amp;
            amp *= 0.5f;
            freq *= 2.0f;
        }
        return norm > 0.0f ? sum / norm : 0.0f;
    }

    private static uint SliceSeed(uint seed, long slice) =>
        seed + unchecked((uint)slice * TimePrime);

    /// <summary>Time-evolving 2D fBm in [-1, 1].</summary>
    public static float Fbm2(float x, float y, float t, uint seed, int octaves)
    {
        int oct = System.Math.Clamp(octaves, 1, 8);
        float t0 = MathF.Floor(t);
        float ft = Fade(t - t0);
        uint s0 = SliceSeed(seed, (long)t0);
        uint s1 = SliceSeed(seed, (long)t0 + 1);
        return Lerp(Fbm2Static(x, y, s0, oct), Fbm2Static(x, y, s1, oct), ft);
    }

    /// <summary>Time-evolving 3D fBm in [-1, 1].</summary>
    public static float Fbm3(float x, float y, float z, float t, uint seed, int octaves)
    {
        int oct = System.Math.Clamp(octaves, 1, 8);
        float t0 = MathF.Floor(t);
        float ft = Fade(t - t0);
        uint s0 = SliceSeed(seed, (long)t0);
        uint s1 = SliceSeed(seed, (long)t0 + 1);
        return Lerp(Fbm3Static(x, y, z, s0, oct), Fbm3Static(x, y, z, s1, oct), ft);
    }

    /// <summary>Divergence-free 2D flow via the curl of a scalar potential.</summary>
    public static Vector2 Curl2(float x, float y, float t, uint seed, int octaves)
    {
        const float e = 1e-3f;
        float dpdy = (Fbm2(x, y + e, t, seed, octaves) - Fbm2(x, y - e, t, seed, octaves)) / (2.0f * e);
        float dpdx = (Fbm2(x + e, y, t, seed, octaves) - Fbm2(x - e, y, t, seed, octaves)) / (2.0f * e);
        return new Vector2(dpdy, -dpdx);
    }

    /// <summary>Divergence-free 3D flow via the curl of a vector potential.</summary>
    public static Vector3 Curl3(float x, float y, float z, float t, uint seed, int octaves)
    {
        const float e = 1e-3f;
        const uint oa = 0x11111111u;
        const uint ob = 0x22222222u;
        const uint oc = 0x33333333u;
        float PsiX(float px, float py, float pz) => Fbm3(px, py, pz, t, seed ^ oa, octaves);
        float PsiY(float px, float py, float pz) => Fbm3(px, py, pz, t, seed ^ ob, octaves);
        float PsiZ(float px, float py, float pz) => Fbm3(px, py, pz, t, seed ^ oc, octaves);
        float Ddy(Func<float, float, float, float> f) => (f(x, y + e, z) - f(x, y - e, z)) / (2.0f * e);
        float Ddz(Func<float, float, float, float> f) => (f(x, y, z + e) - f(x, y, z - e)) / (2.0f * e);
        float Ddx(Func<float, float, float, float> f) => (f(x + e, y, z) - f(x - e, y, z)) / (2.0f * e);
        return new Vector3(
            Ddy(PsiZ) - Ddz(PsiY),
            Ddx(PsiX) - Ddz(PsiZ),
            Ddx(PsiY) - Ddy(PsiX));
    }
}
