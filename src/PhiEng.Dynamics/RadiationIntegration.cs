using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>
/// Radiative heat transfer (2D and 3D).
/// </summary>
/// <remarks>
/// <para>
/// Managed mirror of the Rust radiation module
/// (<c>crates/dynamics/src/radiation.rs</c>). The thermal phase calls
/// into this class after integration, so it sees the final positions of
/// the frame.
/// </para>
/// <para>
/// The model is the standard gray-body exchange:
/// body to environment is
/// <c>Q = eps * sigma * A * (T^4 - T_amb^4)</c> (Stefan-Boltzmann), and
/// body to body is
/// <c>Q = eps_eff * sigma * A * F * (T_i^4 - T_j^4)</c> with the
/// two-gray-body effective emissivity
/// <c>eps_eff = 1 / (1/eps_i + 1/eps_j - 1)</c> and the view factor
/// <c>F = min((r_j / d)^2, 1)</c>.
/// </para>
/// <para>
/// The 2D and 3D variants deliberately differ: 2D treats a body's
/// "surface area" as its perimeter and uses the <i>characteristic
/// length</i> in the pair term, while 3D uses the true surface area.
/// The asymmetry is reproduced verbatim so the managed mirror matches
/// the Rust core.
/// </para>
/// </remarks>
public static class RadiationIntegration
{
    internal static void ComputeRadiation2(
        List<RigidBody2> bodies,
        ThermalConfig2 config,
        float[] heatFlux,
        float[] surfaceArea)
    {
        float sigma = config.StefanBoltzmann;
        float tAmb = config.AmbientTemperature;
        float tAmbSq = tAmb * tAmb;
        float tAmbFourth = tAmbSq * tAmbSq;
        float cutoffFactor = config.RadiationRadiusFactor;

        for (int i = 0; i < bodies.Count; i++)
        {
            var bodyI = bodies[i];
            if (bodyI.Shape == null) continue;

            float epsI = System.Math.Clamp(bodyI.Material.Emissivity, 0.0f, 1.0f);
            if (epsI <= 0.0f) continue;

            float aSurf = surfaceArea[i];
            if (aSurf <= 0.0f) continue;

            float tI = System.Math.Clamp(bodyI.Temperature, 0.0f, config.MaxTemperature);
            float tISq = tI * tI;
            float tIFourth = tISq * tISq;

            // q_env is the radiative loss (positive when hotter than
            // ambient), but heat_flux accumulates heat gained.
            float qEnv = epsI * sigma * aSurf * (tIFourth - tAmbFourth);
            heatFlux[i] -= qEnv;

            float rI = ThermalGeometry.BoundingRadius2(bodyI.Shape);
            float lI = ThermalGeometry.CharacteristicLength2(bodyI.Shape);

            for (int j = i + 1; j < bodies.Count; j++)
            {
                var bodyJ = bodies[j];
                if (bodyJ.Shape == null) continue;

                float epsJ = System.Math.Clamp(bodyJ.Material.Emissivity, 0.0f, 1.0f);
                if (epsJ <= 0.0f) continue;

                float rJ = ThermalGeometry.BoundingRadius2(bodyJ.Shape);
                float dist = (bodyI.Transform.Position - bodyJ.Transform.Position).Length();
                float cutoff = cutoffFactor * (rI + rJ);
                if (dist > cutoff || dist <= 0.0f) continue;

                float epsEff = 1.0f / (1.0f / epsI + 1.0f / epsJ - 1.0f);
                if (epsEff <= 0.0f) continue;

                float tJ = System.Math.Clamp(bodyJ.Temperature, 0.0f, config.MaxTemperature);
                float tJSq = tJ * tJ;
                float tJFourth = tJSq * tJSq;

                float viewFactor = MathF.Min((rJ / dist) * (rJ / dist), 1.0f);

                // Occlusion: a third body between the pair blocks the
                // line of sight. Opt-in: it costs an O(n) scan per pair.
                if (config.RadiationOcclusion && IsOccluded2(bodies, i, j)) continue;

                // Note: the 2D pair term uses the characteristic length
                // (the 3D one uses the surface area) — reproduced
                // verbatim from the Rust core.
                float qRad = epsEff * sigma * lI * viewFactor * (tIFourth - tJFourth);

                heatFlux[i] -= qRad;
                heatFlux[j] += qRad;
            }
        }
    }

    internal static void ComputeRadiation3(
        List<RigidBody3> bodies,
        ThermalConfig2 config,
        float[] heatFlux,
        float[] surfaceArea)
    {
        float sigma = config.StefanBoltzmann;
        float tAmb = config.AmbientTemperature;
        float tAmbSq = tAmb * tAmb;
        float tAmbFourth = tAmbSq * tAmbSq;
        float cutoffFactor = config.RadiationRadiusFactor;

        for (int i = 0; i < bodies.Count; i++)
        {
            var bodyI = bodies[i];
            if (bodyI.Shape == null) continue;

            float epsI = System.Math.Clamp(bodyI.Material.Emissivity, 0.0f, 1.0f);
            if (epsI <= 0.0f) continue;

            float aSurf = surfaceArea[i];
            if (aSurf <= 0.0f) continue;

            float tI = System.Math.Clamp(bodyI.Temperature, 0.0f, config.MaxTemperature);
            float tISq = tI * tI;
            float tIFourth = tISq * tISq;

            float qEnv = epsI * sigma * aSurf * (tIFourth - tAmbFourth);
            heatFlux[i] -= qEnv;

            float rI = ThermalGeometry.BoundingRadius3(bodyI.Shape);

            for (int j = i + 1; j < bodies.Count; j++)
            {
                var bodyJ = bodies[j];
                if (bodyJ.Shape == null) continue;

                float epsJ = System.Math.Clamp(bodyJ.Material.Emissivity, 0.0f, 1.0f);
                if (epsJ <= 0.0f) continue;

                float rJ = ThermalGeometry.BoundingRadius3(bodyJ.Shape);
                float dist = (bodyI.Transform.Position - bodyJ.Transform.Position).Length();
                float cutoff = cutoffFactor * (rI + rJ);
                if (dist > cutoff || dist <= 0.0f) continue;

                float epsEff = 1.0f / (1.0f / epsI + 1.0f / epsJ - 1.0f);
                if (epsEff <= 0.0f) continue;

                float tJ = System.Math.Clamp(bodyJ.Temperature, 0.0f, config.MaxTemperature);
                float tJSq = tJ * tJ;
                float tJFourth = tJSq * tJSq;

                float viewFactor = MathF.Min((rJ / dist) * (rJ / dist), 1.0f);

                if (config.RadiationOcclusion && IsOccluded3(bodies, i, j)) continue;

                // The 3D pair term uses the surface area (the 2D one uses
                // the characteristic length).
                float qRad = epsEff * sigma * aSurf * viewFactor * (tIFourth - tJFourth);

                heatFlux[i] -= qRad;
                heatFlux[j] += qRad;
            }
        }
    }

    public static bool IsOccluded2(List<RigidBody2> bodies, int a, int b)
    {
        Vector2 pa = bodies[a].Transform.Position;
        Vector2 pb = bodies[b].Transform.Position;
        Vector2 ab = pb - pa;
        float lenSq = ab.LengthSquared();
        if (lenSq <= 0.0f) return false;
        Vector2 mid = (pa + pb) * 0.5f;
        float half = MathF.Sqrt(lenSq) * 0.5f;

        for (int idx = 0; idx < bodies.Count; idx++)
        {
            if (idx == a || idx == b) continue;
            var body = bodies[idx];
            if (body.Shape == null) continue;
            Vector2 p = body.Transform.Position;
            // Parameter along the segment; must be strictly inside.
            float t = Vector2.Dot(p - pa, ab) / lenSq;
            if (!(t > 0.0f && t < 1.0f)) continue;
            // Distance from the segment.
            Vector2 closest = pa + ab * t;
            float d = (p - closest).Length();
            float radius = ThermalGeometry.BoundingRadius2(body.Shape);
            // The blocker must reach the line (within its radius) and sit
            // closer to the midpoint than the endpoints do.
            if (d <= radius && (p - mid).Length() < half) return true;
        }
        return false;
    }

    public static bool IsOccluded3(List<RigidBody3> bodies, int a, int b)
    {
        Vector3 pa = bodies[a].Transform.Position;
        Vector3 pb = bodies[b].Transform.Position;
        Vector3 ab = pb - pa;
        float lenSq = ab.LengthSquared();
        if (lenSq <= 0.0f) return false;
        Vector3 mid = (pa + pb) * 0.5f;
        float half = MathF.Sqrt(lenSq) * 0.5f;

        for (int idx = 0; idx < bodies.Count; idx++)
        {
            if (idx == a || idx == b) continue;
            var body = bodies[idx];
            if (body.Shape == null) continue;
            Vector3 p = body.Transform.Position;
            float t = Vector3.Dot(p - pa, ab) / lenSq;
            if (!(t > 0.0f && t < 1.0f)) continue;
            Vector3 closest = pa + ab * t;
            float d = (p - closest).Length();
            float radius = ThermalGeometry.BoundingRadius3(body.Shape);
            if (d <= radius && (p - mid).Length() < half) return true;
        }
        return false;
    }

}
