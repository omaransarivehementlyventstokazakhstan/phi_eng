using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>
/// Per-world thermal scratch state. Mirrors the Rust
/// <c>ThermalState2</c> (the 3D world uses the same type, as in Rust).
/// </summary>
public sealed class ThermalState2
{
    /// <summary>Thermal tunables for this world.</summary>
    public ThermalConfig2 Config { get; set; } = new();

    /// <summary>Per-body accumulated heat gain (watts) for this step.</summary>
    internal float[] HeatFlux = Array.Empty<float>();

    /// <summary>Per-body cached surface area for this step.</summary>
    internal float[] SurfaceArea = Array.Empty<float>();
}

/// <summary>
/// Shape geometry helpers used by the thermal subsystem. Mirrors the
/// Rust <c>Shape2::surface_area</c> / <c>characteristic_length</c> /
/// <c>bounding_radius</c> / <c>scaled</c> and their 3D counterparts.
/// </summary>
/// <remarks>
/// The 2D and 3D definitions deliberately differ (2D "surface area" is a
/// perimeter, 2D characteristic length is <c>sqrt(area)</c> while 3D is
/// <c>cbrt(volume)</c>, the polygon centroid is area-based while the hull
/// centroid is the vertex mean). These asymmetries are reproduced
/// verbatim so the managed mirror matches the Rust core.
/// </remarks>
public static class ThermalGeometry
{
    // ------------------------------------------------------------------
    // 2D
    // ------------------------------------------------------------------

    /// <summary>Surface area of a 2D shape (a perimeter in 2D).</summary>
    public static float SurfaceArea2(Shape2? shape)
    {
        switch (shape)
        {
            case CircleShape c:
                return 2.0f * MathF.PI * c.Radius;
            case BoxShape2 b:
                return 4.0f * (b.HalfExtents.X + b.HalfExtents.Y);
            case PolygonShape p:
            {
                float perimeter = 0.0f;
                for (int i = 0; i < p.Vertices.Count; i++)
                {
                    int j = (i + 1) % p.Vertices.Count;
                    perimeter += (p.Vertices[j] - p.Vertices[i]).Length();
                }
                return perimeter;
            }
            default:
                return 0.0f;
        }
    }

    /// <summary>Characteristic length scale of a 2D shape: sqrt(area).</summary>
    public static float CharacteristicLength2(Shape2? shape)
    {
        switch (shape)
        {
            case CircleShape c:
                return c.Radius;
            case BoxShape2 b:
                return MathF.Sqrt(4.0f * b.HalfExtents.X * b.HalfExtents.Y);
            case PolygonShape p:
            {
                var (area, _) = RigidBody2.PolygonAreaAndCentroid(p.Vertices);
                return MathF.Sqrt(MathF.Abs(area));
            }
            default:
                return 0.0f;
        }
    }

    /// <summary>Bounding radius of a 2D shape (max vertex distance from the centroid).</summary>
    public static float BoundingRadius2(Shape2? shape)
    {
        switch (shape)
        {
            case CircleShape c:
                return c.Radius;
            case BoxShape2 b:
                return b.HalfExtents.Length();
            case PolygonShape p:
            {
                var (_, centroid) = RigidBody2.PolygonAreaAndCentroid(p.Vertices);
                float max = 0.0f;
                foreach (var v in p.Vertices)
                {
                    float d = (v - centroid).Length();
                    if (d > max) max = d;
                }
                return max;
            }
            default:
                return 0.0f;
        }
    }

    /// <summary>
    /// A uniformly scaled copy of a 2D shape (thermal expansion).
    /// A non-positive or non-finite factor returns the shape unchanged.
    /// </summary>
    public static Shape2? Scaled2(Shape2? shape, float factor)
    {
        if (shape == null) return null;
        if (!float.IsFinite(factor) || factor <= 0.0f) return shape;
        switch (shape)
        {
            case CircleShape c:
                return new CircleShape(c.Radius * factor);
            case BoxShape2 b:
                return new BoxShape2(b.HalfExtents * factor);
            case PolygonShape p:
            {
                var scaled = new PolygonShape(new List<Vector2>());
                foreach (var v in p.Vertices) scaled.Vertices.Add(v * factor);
                return scaled;
            }
            default:
                return shape;
        }
    }

    // ------------------------------------------------------------------
    // 3D
    // ------------------------------------------------------------------

    /// <summary>Surface area of a 3D shape.</summary>
    public static float SurfaceArea3(Shape3? shape)
    {
        switch (shape)
        {
            case SphereShape3 s:
                return 4.0f * MathF.PI * s.Radius * s.Radius;
            case BoxShape3 b:
            {
                float w = b.HalfExtents.X * 2.0f;
                float h = b.HalfExtents.Y * 2.0f;
                float d = b.HalfExtents.Z * 2.0f;
                return 2.0f * (w * h + w * d + h * d);
            }
            case CapsuleShape3 c:
                return 4.0f * MathF.PI * c.Radius * c.Radius
                    + 2.0f * MathF.PI * c.Radius * 2.0f * c.HalfLength;
            case ConvexHullShape3 hull:
            {
                float area = 0.0f;
                var v = hull.Vertices;
                for (int i = 0; i < v.Count; i++)
                {
                    for (int j = i + 1; j < v.Count; j++)
                    {
                        for (int k = j + 1; k < v.Count; k++)
                        {
                            var a = v[i];
                            var b = v[j];
                            var c2 = v[k];
                            area += 0.5f * Vector3.Cross(b - a, c2 - a).Length();
                        }
                    }
                }
                return area;
            }
            default:
                return 0.0f;
        }
    }

    /// <summary>Characteristic length of a 3D shape: cbrt(volume).</summary>
    public static float CharacteristicLength3(Shape3? shape)
    {
        switch (shape)
        {
            case SphereShape3 s:
                return s.Radius;
            case BoxShape3 b:
            {
                float vol = 8.0f * b.HalfExtents.X * b.HalfExtents.Y * b.HalfExtents.Z;
                return MathF.Cbrt(vol);
            }
            case CapsuleShape3 c:
            {
                float vol = MathF.PI * c.Radius * c.Radius * 2.0f * c.HalfLength
                    + (4.0f / 3.0f) * MathF.PI * c.Radius * c.Radius * c.Radius;
                return MathF.Cbrt(vol);
            }
            case ConvexHullShape3 hull:
            {
                double vol = 0.0;
                var v = hull.Vertices;
                for (int i = 0; i < v.Count; i++)
                {
                    for (int j = i + 1; j < v.Count; j++)
                    {
                        for (int k = j + 1; k < v.Count; k++)
                        {
                            var a = v[i];
                            var b = v[j];
                            var c2 = v[k];
                            vol += (double)(Vector3.Dot(a, Vector3.Cross(b, c2)) / 6.0f);
                        }
                    }
                }
                return MathF.Cbrt((float)System.Math.Abs(vol));
            }
            default:
                return 0.0f;
        }
    }

    /// <summary>Bounding radius of a 3D shape (max vertex distance from the centroid).</summary>
    public static float BoundingRadius3(Shape3? shape)
    {
        switch (shape)
        {
            case SphereShape3 s:
                return s.Radius;
            case BoxShape3 b:
                return b.HalfExtents.Length();
            case CapsuleShape3 c:
                return c.HalfLength + c.Radius;
            case ConvexHullShape3 hull:
            {
                var v = hull.Vertices;
                if (v.Count == 0) return 0.0f;
                Vector3 centroid = Vector3.Zero;
                foreach (var p in v) centroid += p;
                centroid /= v.Count;
                float max = 0.0f;
                foreach (var p in v)
                {
                    float d = (p - centroid).Length();
                    if (d > max) max = d;
                }
                return max;
            }
            default:
                return 0.0f;
        }
    }

    /// <summary>
    /// A uniformly scaled copy of a 3D shape (thermal expansion).
    /// A non-positive or non-finite factor returns the shape unchanged.
    /// </summary>
    public static Shape3? Scaled3(Shape3? shape, float factor)
    {
        if (shape == null) return null;
        if (!float.IsFinite(factor) || factor <= 0.0f) return shape;
        switch (shape)
        {
            case SphereShape3 s:
                return new SphereShape3(s.Radius * factor);
            case BoxShape3 b:
                return new BoxShape3(b.HalfExtents * factor);
            case CapsuleShape3 c:
                return new CapsuleShape3(c.Radius * factor, c.HalfLength * factor);
            case ConvexHullShape3 hull:
            {
                var scaled = new ConvexHullShape3(new List<Vector3>());
                foreach (var v in hull.Vertices) scaled.Vertices.Add(v * factor);
                return scaled;
            }
            default:
                return shape;
        }
    }
}

/// <summary>
/// Managed mirror of the Rust thermal phase. Mirrors
/// <c>crates/dynamics/src/thermal.rs</c>: conduction, radiation and
/// convection, plus the opt-in radiation-occlusion, forced-convection and
/// thermal-expansion sub-features.
/// </summary>
/// <remarks>
/// The phase runs immediately after velocity integration and before the
/// sleep check, exactly as in the Rust core. It does not modify
/// velocity, so it neither wakes sleeping bodies nor is affected by
/// sleep.
/// </remarks>
public static class ThermalIntegration
{
    /// <summary>
    /// Target linear scale factor for a body at <paramref name="temperature"/>,
    /// given the expansion coefficient and reference temperature.
    /// </summary>
    public static float ExpansionFactor(ThermalConfig2 config, float temperature)
    {
        float alpha = config.ExpansionCoefficient;
        if (!float.IsFinite(alpha) || alpha == 0.0f) return 1.0f;
        float dt = temperature - config.ExpansionReferenceTemperature;
        if (!float.IsFinite(dt)) return 1.0f;
        float factor = 1.0f + alpha * dt;
        // A non-positive or non-finite factor would produce a degenerate
        // shape; clamp instead.
        if (!float.IsFinite(factor) || factor <= 0.0f) return 1.0f;
        return System.Math.Clamp(factor, 0.01f, 100.0f);
    }

    // ------------------------------------------------------------------
    // 2D phase
    // ------------------------------------------------------------------

    /// <summary>
    /// Run the 2D thermal phase: conduction, radiation, convection, then
    /// the temperature update.
    /// </summary>
    public static void StepThermal2(
        List<RigidBody2> bodies,
        List<ContactConstraint> contacts,
        ThermalConfig2 config,
        ThermalState2 state,
        float dt)
    {
        if (!config.IsActive() || dt <= 0.0f || bodies.Count == 0) return;

        int n = bodies.Count;
        if (state.HeatFlux.Length != n)
        {
            Array.Resize(ref state.HeatFlux, n);
            Array.Resize(ref state.SurfaceArea, n);
        }

        ResetThermalScratch(bodies, state);

        if (config.ConductionEnabled) ComputeConduction2(bodies, contacts, state);
        if (config.RadiationEnabled)
            RadiationIntegration.ComputeRadiation2(bodies, config, state.HeatFlux, state.SurfaceArea);
        if (config.ConvectionEnabled) ComputeConvection2(bodies, config, state);

        UpdateTemperatures2(bodies, config, state, dt);
    }

    /// <summary>
    /// Apply thermal expansion to 2D bodies: rescale each shape to match
    /// its expansion factor and refresh the mass properties. Off by
    /// default, in which case this is a no-op.
    /// </summary>
    public static void ApplyExpansion2(List<RigidBody2> bodies, ThermalConfig2 config)
    {
        if (!float.IsFinite(config.ExpansionCoefficient) || config.ExpansionCoefficient == 0.0f)
            return;
        foreach (var body in bodies)
        {
            float target = ExpansionFactor(config, body.Temperature);
            RescaleForExpansion2(body, target);
        }
    }

    private static void ResetThermalScratch(List<RigidBody2> bodies, ThermalState2 state)
    {
        for (int i = 0; i < bodies.Count; i++)
        {
            state.HeatFlux[i] = 0.0f;
            state.SurfaceArea[i] = ThermalGeometry.SurfaceArea2(bodies[i].Shape);
        }
    }

    private static void ComputeConduction2(
        List<RigidBody2> bodies,
        List<ContactConstraint> contacts,
        ThermalState2 state)
    {
        foreach (var contact in contacts)
        {
            int a = contact.BodyA;
            int b = contact.BodyB;
            if (a >= bodies.Count || b >= bodies.Count) continue;

            var bodyA = bodies[a];
            var bodyB = bodies[b];

            float kA = bodyA.Material.ThermalConductivity;
            float kB = bodyB.Material.ThermalConductivity;
            if (kA <= 0.0f || kB <= 0.0f) continue;

            float kEff = (kA * kB) / (kA + kB);

            float charA = bodyA.Shape != null
                ? ThermalGeometry.CharacteristicLength2(bodyA.Shape)
                : ThermalConfig2.MinConductionDistance;
            float charB = bodyB.Shape != null
                ? ThermalGeometry.CharacteristicLength2(bodyB.Shape)
                : ThermalConfig2.MinConductionDistance;
            float wContact = MathF.Max(MathF.Min(charA, charB), ThermalConfig2.MinConductionDistance);

            float gCond = ThermalConfig2.ConductionScale * kEff * wContact;

            float qDot = gCond * (bodyA.Temperature - bodyB.Temperature);

            state.HeatFlux[a] -= qDot;
            state.HeatFlux[b] += qDot;
        }
    }

    private static void ComputeConvection2(
        List<RigidBody2> bodies,
        ThermalConfig2 config,
        ThermalState2 state)
    {
        float h = config.ConvectionCoefficient;
        float tAmb = config.AmbientTemperature;
        float hForced = config.ForcedConvectionCoefficient;

        for (int i = 0; i < bodies.Count; i++)
        {
            var body = bodies[i];
            if (body.Shape == null) continue;
            float aSurf = state.SurfaceArea[i];
            if (aSurf <= 0.0f) continue;
            // Forced convection: the coefficient grows with the surface
            // speed (a body moving through still air sheds heat faster).
            float hEff = hForced > 0.0f ? h + hForced * body.Velocity.Length() : h;
            float qConv = hEff * aSurf * (body.Temperature - tAmb);
            state.HeatFlux[i] -= qConv;
        }
    }

    private static void UpdateTemperatures2(
        List<RigidBody2> bodies,
        ThermalConfig2 config,
        ThermalState2 state,
        float dt)
    {
        float maxDelta = config.MaxTemperatureRate * dt;
        float minT = config.MinTemperature;
        float maxT = config.MaxTemperature;

        for (int i = 0; i < bodies.Count; i++)
        {
            var body = bodies[i];
            // Statics are infinite-capacity reservoirs: they exchange heat
            // but never change temperature.
            if (body.IsStatic) continue;

            float cI = body.Mass * body.Material.SpecificHeat;
            if (cI <= 0.0f) continue;

            float deltaT = System.Math.Clamp((state.HeatFlux[i] / cI) * dt, -maxDelta, maxDelta);
            body.Temperature = System.Math.Clamp(body.Temperature + deltaT, minT, maxT);
        }
    }

    /// <summary>
    /// True when a third body lies between bodies <paramref name="a"/> and
    /// <paramref name="b"/>, blocking the line of sight (2D).
    /// </summary>
    private static void RescaleForExpansion2(RigidBody2 body, float target)
    {
        float applied = body.ExpansionScale;
        if (!float.IsFinite(applied) || applied <= 0.0f)
        {
            // Defensive: an invalid tracker would make the ratio NaN.
            body.ExpansionScale = 1.0f;
            return;
        }
        if (MathF.Abs(target - applied) <= float.Epsilon) return;
        if (body.Shape == null) return;
        float ratio = target / applied;
        if (!float.IsFinite(ratio) || ratio <= 0.0f) return;
        var scaled = ThermalGeometry.Scaled2(body.Shape, ratio);
        if (scaled == null) return;
        float density = body.Material.Density;
        body.SetShape(scaled, density);
        body.ExpansionScale = target;
    }

    // ------------------------------------------------------------------
    // 3D phase
    // ------------------------------------------------------------------

    /// <summary>
    /// Run the 3D thermal phase: conduction, radiation, convection, then
    /// the temperature update.
    /// </summary>
    public static void StepThermal3(
        List<RigidBody3> bodies,
        List<ContactConstraint3D> contacts,
        ThermalConfig2 config,
        ThermalState2 state,
        float dt)
    {
        if (!config.IsActive() || dt <= 0.0f || bodies.Count == 0) return;

        int n = bodies.Count;
        if (state.HeatFlux.Length != n)
        {
            Array.Resize(ref state.HeatFlux, n);
            Array.Resize(ref state.SurfaceArea, n);
        }

        ResetThermalScratch3(bodies, state);

        if (config.ConductionEnabled) ComputeConduction3(bodies, contacts, state);
        if (config.RadiationEnabled)
            RadiationIntegration.ComputeRadiation3(bodies, config, state.HeatFlux, state.SurfaceArea);
        if (config.ConvectionEnabled) ComputeConvection3(bodies, config, state);

        UpdateTemperatures3(bodies, config, state, dt);
    }

    /// <summary>
    /// Apply thermal expansion to 3D bodies. See
    /// <see cref="ApplyExpansion2"/>.
    /// </summary>
    public static void ApplyExpansion3(List<RigidBody3> bodies, ThermalConfig2 config)
    {
        if (!float.IsFinite(config.ExpansionCoefficient) || config.ExpansionCoefficient == 0.0f)
            return;
        foreach (var body in bodies)
        {
            float target = ExpansionFactor(config, body.Temperature);
            RescaleForExpansion3(body, target);
        }
    }

    private static void ResetThermalScratch3(List<RigidBody3> bodies, ThermalState2 state)
    {
        for (int i = 0; i < bodies.Count; i++)
        {
            state.HeatFlux[i] = 0.0f;
            state.SurfaceArea[i] = ThermalGeometry.SurfaceArea3(bodies[i].Shape);
        }
    }

    private static void ComputeConduction3(
        List<RigidBody3> bodies,
        List<ContactConstraint3D> contacts,
        ThermalState2 state)
    {
        foreach (var contact in contacts)
        {
            int a = contact.BodyA;
            int b = contact.BodyB;
            if (a >= bodies.Count || b >= bodies.Count) continue;

            var bodyA = bodies[a];
            var bodyB = bodies[b];

            float kA = bodyA.Material.ThermalConductivity;
            float kB = bodyB.Material.ThermalConductivity;
            if (kA <= 0.0f || kB <= 0.0f) continue;

            float kEff = (kA * kB) / (kA + kB);

            float rA = bodyA.Shape != null
                ? ThermalGeometry.BoundingRadius3(bodyA.Shape)
                : ThermalConfig2.MinConductionDistance;
            float rB = bodyB.Shape != null
                ? ThermalGeometry.BoundingRadius3(bodyB.Shape)
                : ThermalConfig2.MinConductionDistance;
            float rMin = MathF.Max(MathF.Min(rA, rB), ThermalConfig2.MinConductionDistance);

            float dPen = MathF.Max(contact.Penetration, 0.0f);
            float aContact = MathF.PI * rMin * MathF.Max(dPen, 0.01f * rMin);
            float dChar = rMin * 0.5f;

            float gCond = ThermalConfig2.ConductionScale * kEff * aContact / dChar;

            float qDot = gCond * (bodyA.Temperature - bodyB.Temperature);

            state.HeatFlux[a] -= qDot;
            state.HeatFlux[b] += qDot;
        }
    }

    private static void ComputeConvection3(
        List<RigidBody3> bodies,
        ThermalConfig2 config,
        ThermalState2 state)
    {
        float h = config.ConvectionCoefficient;
        float tAmb = config.AmbientTemperature;
        float hForced = config.ForcedConvectionCoefficient;

        for (int i = 0; i < bodies.Count; i++)
        {
            var body = bodies[i];
            if (body.Shape == null) continue;
            float aSurf = state.SurfaceArea[i];
            if (aSurf <= 0.0f) continue;
            float hEff = hForced > 0.0f ? h + hForced * body.Velocity.Length() : h;
            float qConv = hEff * aSurf * (body.Temperature - tAmb);
            state.HeatFlux[i] -= qConv;
        }
    }

    private static void UpdateTemperatures3(
        List<RigidBody3> bodies,
        ThermalConfig2 config,
        ThermalState2 state,
        float dt)
    {
        float maxDelta = config.MaxTemperatureRate * dt;
        float minT = config.MinTemperature;
        float maxT = config.MaxTemperature;

        for (int i = 0; i < bodies.Count; i++)
        {
            var body = bodies[i];
            if (body.IsStatic) continue;

            float cI = body.Mass * body.Material.SpecificHeat;
            if (cI <= 0.0f) continue;

            float deltaT = System.Math.Clamp((state.HeatFlux[i] / cI) * dt, -maxDelta, maxDelta);
            body.Temperature = System.Math.Clamp(body.Temperature + deltaT, minT, maxT);
        }
    }

    /// <summary>
    /// True when a third body lies between bodies <paramref name="a"/> and
    /// <paramref name="b"/>, blocking the line of sight (3D).
    /// </summary>
    private static void RescaleForExpansion3(RigidBody3 body, float target)
    {
        float applied = body.ExpansionScale;
        if (!float.IsFinite(applied) || applied <= 0.0f)
        {
            body.ExpansionScale = 1.0f;
            return;
        }
        if (MathF.Abs(target - applied) <= float.Epsilon) return;
        if (body.Shape == null) return;
        float ratio = target / applied;
        if (!float.IsFinite(ratio) || ratio <= 0.0f) return;
        var scaled = ThermalGeometry.Scaled3(body.Shape, ratio);
        if (scaled == null) return;
        float density = body.Material.Density;
        body.SetShape(scaled, density);
        body.ExpansionScale = target;
    }
}
