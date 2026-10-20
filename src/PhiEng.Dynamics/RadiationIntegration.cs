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

        // Sky / ground environment split. A non-positive sky or ground
        // temperature means "use the ambient". When the two end up equal
        // (the default) the historical single-temperature exchange is
        // used, so the result stays bit-identical.
        (float tSky, float tGround) = EnvironmentTemperatures(config);
        // The split applies whenever the configured temperatures differ
        // from the ambient: a sky and a ground that are equal to *each
        // other* but not to the ambient must still be honoured. When both
        // resolve to the ambient, the historical single-temperature
        // exchange is used and the result is bit-identical to it.
        bool splitEnvironment = tSky != tGround || tSky != tAmb;
        float tSkyFourth = FourthPower(tSky);
        float tGroundFourth = FourthPower(tGround);

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

            // Fraction of the body's surface that faces the sky.
            float fUp = ThermalGeometry.UpFacingFraction2(bodyI.Shape);

            // q_env is the radiative loss (positive when hotter than
            // ambient), but heat_flux accumulates heat gained.
            float qEnv = splitEnvironment
                ? epsI * sigma * aSurf
                    * (fUp * (tIFourth - tSkyFourth) + (1.0f - fUp) * (tIFourth - tGroundFourth))
                : epsI * sigma * aSurf * (tIFourth - tAmbFourth);
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
                float lJ = ThermalGeometry.CharacteristicLength2(bodyJ.Shape);
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
                bool occluded = config.RadiationOcclusion && IsOccluded2(bodies, i, j);

                // Note: the 2D pair term uses the characteristic length
                // (the 3D one uses the surface area) — reproduced
                // verbatim from the Rust core.
                float qRad;
                if (config.ViewFactorModel == ViewFactorModel.Symmetric)
                {
                    // Order-independent view factor: the geometric mean
                    // of the two directed factors, so the exchange rate
                    // does not depend on which body the pair loop
                    // happened to visit first.
                    float f = System.Math.Min((rI * rJ) / (dist * dist), 1.0f);
                    // The mean of the two emitting areas; for two equal
                    // bodies this is exactly the shared area.
                    float aEff = 0.5f * (lI + lJ) * f;
                    qRad = epsEff * sigma * aEff * (tIFourth - tJFourth);
                    if (occluded)
                    {
                        // A partially transparent occluder attenuates the
                        // exchange instead of blocking it completely.
                        qRad *= config.OccluderTransmittance;
                    }
                }
                else
                {
                    if (occluded)
                    {
                        if (config.OccluderTransmittance <= 0.0f)
                        {
                            // Fully opaque: the historical behaviour.
                            continue;
                        }
                        qRad = epsEff * sigma * lI * viewFactor * (tIFourth - tJFourth)
                            * config.OccluderTransmittance;
                    }
                    else
                    {
                        qRad = epsEff * sigma * lI * viewFactor * (tIFourth - tJFourth);
                    }
                }

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

        // Sky / ground environment split. A non-positive sky or ground
        // temperature means "use the ambient". When the two end up equal
        // (the default) the historical single-temperature exchange is
        // used, so the result stays bit-identical.
        (float tSky, float tGround) = EnvironmentTemperatures(config);
        // The split applies whenever the configured temperatures differ
        // from the ambient: a sky and a ground that are equal to *each
        // other* but not to the ambient must still be honoured. When both
        // resolve to the ambient, the historical single-temperature
        // exchange is used and the result is bit-identical to it.
        bool splitEnvironment = tSky != tGround || tSky != tAmb;
        float tSkyFourth = FourthPower(tSky);
        float tGroundFourth = FourthPower(tGround);

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

            // Fraction of the body's surface that faces the sky.
            float fUp = ThermalGeometry.UpFacingFraction3(bodyI.Shape);

            float qEnv = splitEnvironment
                ? epsI * sigma * aSurf
                    * (fUp * (tIFourth - tSkyFourth) + (1.0f - fUp) * (tIFourth - tGroundFourth))
                : epsI * sigma * aSurf * (tIFourth - tAmbFourth);
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

                // Occlusion (3D): see IsOccluded2.
                bool occluded = config.RadiationOcclusion && IsOccluded3(bodies, i, j);

                float qRad;
                if (config.ViewFactorModel == ViewFactorModel.Symmetric)
                {
                    // Order-independent view factor (see the 2D
                    // counterpart).
                    float f = System.Math.Min((rI * rJ) / (dist * dist), 1.0f);
                    // The mean of the two emitting areas.
                    float aEff = 0.5f * (aSurf + surfaceArea[j]) * f;
                    qRad = epsEff * sigma * aEff * (tIFourth - tJFourth);
                    if (occluded)
                    {
                        qRad *= config.OccluderTransmittance;
                    }
                }
                else
                {
                    if (occluded)
                    {
                        if (config.OccluderTransmittance <= 0.0f)
                        {
                            // Fully opaque: the historical behaviour.
                            continue;
                        }
                        qRad = epsEff * sigma * aSurf * viewFactor * (tIFourth - tJFourth)
                            * config.OccluderTransmittance;
                    }
                    else
                    {
                        qRad = epsEff * sigma * aSurf * viewFactor * (tIFourth - tJFourth);
                    }
                }

                heatFlux[i] -= qRad;
                heatFlux[j] += qRad;
            }
        }
    }

    /// <summary>
    /// True when a third body lies between bodies a and b (2D), blocking
    /// the line of sight.
    /// </summary>
    /// <remarks>
    /// The blocker must project strictly inside the segment (so the two
    /// endpoints never block each other) and reach the line of sight.
    /// The historical implementation additionally required the blocker to
    /// be closer to the segment's midpoint than the endpoints are. That
    /// test was wrong: a blocker sitting just next to one of the two
    /// bodies blocks the line of sight just as effectively as one in the
    /// middle, so the exemption let radiation leak through gaps next to
    /// the endpoints.
    /// </remarks>
    public static bool IsOccluded2(List<RigidBody2> bodies, int a, int b)
    {
        Vector2 pa = bodies[a].Transform.Position;
        Vector2 pb = bodies[b].Transform.Position;
        Vector2 ab = pb - pa;
        float lenSq = ab.LengthSquared();
        if (lenSq <= 0.0f) return false;

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
            // The blocker must reach the line of sight.
            if (d <= radius) return true;
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
            if (d <= radius) return true;
        }
        return false;
    }

    /// <summary>
    /// The sky and ground temperatures used by the environment radiation.
    /// A non-positive value means "use the ambient temperature".
    /// </summary>
    internal static (float Sky, float Ground) EnvironmentTemperatures(ThermalConfig2 config)
    {
        float ambient = config.AmbientTemperature;
        float sky = config.SkyTemperature > 0.0f ? config.SkyTemperature : ambient;
        float ground = config.GroundTemperature > 0.0f ? config.GroundTemperature : ambient;
        return (sky, ground);
    }

    /// <summary>t^4, computed the same way the model has always computed it.</summary>
    internal static float FourthPower(float t)
    {
        float sq = t * t;
        return sq * sq;
    }
}
