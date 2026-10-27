using System.Numerics;

namespace PhiEng.Electromagnetics;

/// <summary>
/// Coulomb pair forces, the electric field and the potential. Mirrors
/// Rust <c>phieng-electromagnetics::electrostatics</c>; the operation order
/// is the same so the two engines agree bit for bit.
/// </summary>
public static class Electrostatics
{
    private const float EpsSq = 1.0e-12f;

    /// <summary>The force on a charge at <paramref name="pi"/> from one at <paramref name="pj"/>.</summary>
    public static Vector2 PairForce2(Vector2 pi, float qi, Vector2 pj, float qj, float k, float softening)
    {
        Vector2 d = pi - pj;
        float r2 = d.LengthSquared() + softening * softening;
        if (r2 < EpsSq || !float.IsFinite(r2)) return Vector2.Zero;
        float s = k * qi * qj;
        if (!float.IsFinite(s)) return Vector2.Zero;
        float inv = 1.0f / MathF.Sqrt(r2);
        return d * (s * inv / r2);
    }

    /// <summary>The 3D counterpart of <see cref="PairForce2"/>.</summary>
    public static Vector3 PairForce3(Vector3 pi, float qi, Vector3 pj, float qj, float k, float softening)
    {
        Vector3 d = pi - pj;
        float r2 = d.LengthSquared() + softening * softening;
        if (r2 < EpsSq || !float.IsFinite(r2)) return Vector3.Zero;
        float s = k * qi * qj;
        if (!float.IsFinite(s)) return Vector3.Zero;
        float inv = 1.0f / MathF.Sqrt(r2);
        return d * (s * inv / r2);
    }

    /// <summary>Fill <paramref name="out"/> with the net Coulomb force on each position.</summary>
    public static void PairwiseForces2(Vector2[] pos, float[] q, float k, float softening, Vector2[] output)
    {
        int n = System.Math.Min(pos.Length, System.Math.Min(q.Length, output.Length));
        for (int i = 0; i < n; i++) output[i] = Vector2.Zero;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                Vector2 f = PairForce2(pos[i], q[i], pos[j], q[j], k, softening);
                output[i] += f;
                output[j] -= f;
            }
        }
    }

    /// <summary>The 3D counterpart of <see cref="PairwiseForces2"/>.</summary>
    public static void PairwiseForces3(Vector3[] pos, float[] q, float k, float softening, Vector3[] output)
    {
        int n = System.Math.Min(pos.Length, System.Math.Min(q.Length, output.Length));
        for (int i = 0; i < n; i++) output[i] = Vector3.Zero;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                Vector3 f = PairForce3(pos[i], q[i], pos[j], q[j], k, softening);
                output[i] += f;
                output[j] -= f;
            }
        }
    }

    /// <summary>The electric field at a point from a set of point charges.</summary>
    public static Vector2 FieldAt2(Vector2 p, Vector2[] pos, float[] q, float k, float softening)
    {
        int n = System.Math.Min(pos.Length, q.Length);
        Vector2 e = Vector2.Zero;
        for (int j = 0; j < n; j++)
        {
            Vector2 d = p - pos[j];
            float r2 = d.LengthSquared() + softening * softening;
            if (r2 < EpsSq || !float.IsFinite(r2)) continue;
            float s = k * q[j];
            if (!float.IsFinite(s)) continue;
            float inv = 1.0f / MathF.Sqrt(r2);
            e += d * (s * inv / r2);
        }
        return e;
    }

    /// <summary>The 3D counterpart of <see cref="FieldAt2"/>.</summary>
    public static Vector3 FieldAt3(Vector3 p, Vector3[] pos, float[] q, float k, float softening)
    {
        int n = System.Math.Min(pos.Length, q.Length);
        Vector3 e = Vector3.Zero;
        for (int j = 0; j < n; j++)
        {
            Vector3 d = p - pos[j];
            float r2 = d.LengthSquared() + softening * softening;
            if (r2 < EpsSq || !float.IsFinite(r2)) continue;
            float s = k * q[j];
            if (!float.IsFinite(s)) continue;
            float inv = 1.0f / MathF.Sqrt(r2);
            e += d * (s * inv / r2);
        }
        return e;
    }

    /// <summary>The electric potential at a point from a set of point charges.</summary>
    public static float PotentialAt2(Vector2 p, Vector2[] pos, float[] q, float k, float softening)
    {
        int n = System.Math.Min(pos.Length, q.Length);
        float v = 0.0f;
        for (int j = 0; j < n; j++)
        {
            Vector2 d = p - pos[j];
            float r2 = d.LengthSquared() + softening * softening;
            if (r2 < EpsSq || !float.IsFinite(r2)) continue;
            float s = k * q[j];
            if (!float.IsFinite(s)) continue;
            v += s / MathF.Sqrt(r2);
        }
        return v;
    }

    /// <summary>The 3D counterpart of <see cref="PotentialAt2"/>.</summary>
    public static float PotentialAt3(Vector3 p, Vector3[] pos, float[] q, float k, float softening)
    {
        int n = System.Math.Min(pos.Length, q.Length);
        float v = 0.0f;
        for (int j = 0; j < n; j++)
        {
            Vector3 d = p - pos[j];
            float r2 = d.LengthSquared() + softening * softening;
            if (r2 < EpsSq || !float.IsFinite(r2)) continue;
            float s = k * q[j];
            if (!float.IsFinite(s)) continue;
            v += s / MathF.Sqrt(r2);
        }
        return v;
    }
}

