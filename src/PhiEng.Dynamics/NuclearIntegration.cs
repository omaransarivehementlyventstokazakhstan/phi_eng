using System.Numerics;
using PhiEng.Math;

namespace PhiEng.Dynamics;

/// <summary>
/// Nuclear physics: radioactive decay, neutron activation, fission and
/// chain reactions. Managed mirror of the Rust nuclear module
/// (crates/dynamics/src/nuclear.rs).
/// </summary>
/// <remarks>
/// <para>
/// The phase runs <i>before</i> the thermal phase. It fills a per-body
/// heat budget (the energy released by decay and fission) and a
/// per-body ionizing-radiation budget, which the thermal phase then
/// turns into temperature. Running it first means the thermal radiation
/// sees the heat in the same step.
/// </para>
/// <para>
/// The decay uses <c>exp</c>, which is <i>not</i> portable across
/// platforms (the libm implementations disagree in the last ULP). It
/// goes through <see cref="DeterministicScalar.Exp"/>, the engine's
/// deterministic exponential, so a half-life drifts by the same amount
/// everywhere and Rust and this mirror agree bit-for-bit.
/// </para>
/// <para>
/// The model is game-scale rather than reactor-scale: it is built from
/// rates, not from particle transport, so it is O(n) per step and
/// deterministic without an RNG.
/// </para>
/// </remarks>
public static class NuclearIntegration
{
    /// <summary>
    /// The minimum of two values, ignoring NaN. This is Rust's
    /// <c>f32::min</c> (IEEE <c>minNum</c>): .NET's <c>Math.Min</c>
    /// *propagates* NaN instead, which would make the two engines
    /// disagree.
    /// </summary>
    internal static float MinNum(float a, float b)
    {
        if (float.IsNaN(a)) return b;
        if (float.IsNaN(b)) return a;
        return System.Math.Min(a, b);
    }

    /// <summary>
    /// The maximum of two values, ignoring NaN - the counterpart of
    /// <see cref="MinNum"/>.
    /// </summary>
    internal static float MaxNum(float a, float b)
    {
        if (float.IsNaN(a)) return b;
        if (float.IsNaN(b)) return a;
        return System.Math.Max(a, b);
    }

    /// <summary>
    /// The product of two values, saturated at <see cref="float.MaxValue"/>
    /// instead of overflowing to infinity. An infinite energy would turn
    /// into NaN in the heat/ionizing split (infinity * 0.0), and a NaN
    /// temperature poisons the body for good.
    /// </summary>
    internal static float SaturatingProduct(float a, float b)
    {
        if (float.IsNaN(a) || float.IsNaN(b))
        {
            return float.NaN;
        }
        float product = a * b;
        if (float.IsInfinity(product))
        {
            return (a < 0.0f) != (b < 0.0f) ? float.NegativeInfinity : float.MaxValue;
        }
        return product;
    }

    /// <summary>
    /// Runs the nuclear phase for a 2D world.
    /// </summary>
    /// <param name="bodies">The world bodies.</param>
    /// <param name="config">The nuclear configuration.</param>
    /// <param name="scratch">The per-step scratch.</param>
    /// <param name="dt">The time step, in simulation seconds.</param>
    public static void StepNuclear2(
        List<RigidBody2> bodies,
        NuclearConfig config,
        NuclearScratch scratch,
        float dt)
    {
        // The scratch is cleared *before* the early returns: the world
        // applies the heat budget after this call, and a stale budget
        // would be re-applied every step (an unbounded temperature
        // ramp).
        int n = bodies.Count;
        if (scratch.Heat.Length != n)
        {
            scratch.Heat = new float[n];
            scratch.Radiation = new float[n];
            scratch.Flux = new float[n];
        }
        else
        {
            Array.Clear(scratch.Heat);
            Array.Clear(scratch.Radiation);
            Array.Clear(scratch.Flux);
        }

        if (!config.IsActive() || dt <= 0.0f || bodies.Count == 0)
        {
            return;
        }
        float dtNuclear = dt * config.TimeScale;
        if (dtNuclear <= 0.0f)
        {
            return;
        }

        // 1. Decay: the activity falls and releases energy.
        for (int i = 0; i < n; i++)
        {
            var ns = bodies[i].Nuclear;
            if (ns.Isotope == 0 || ns.Activity <= 0.0f)
            {
                continue;
            }
            var iso = config.GetIsotope(ns.Isotope);
            if (iso == null)
            {
                continue;
            }
            float lambda = iso.DecayConstant();
            if (lambda <= 0.0f)
            {
                // A stable nuclide never decays.
                ns.Activity = 0.0f;
                continue;
            }
            if (iso.DecayEnergy <= 0.0f)
            {
                continue;
            }

            // The energy released in this step uses the start-of-step
            // activity (explicit, like the thermal phase).
            // The product can overflow to infinity for extreme inputs, and
            // `infinity * 0.0` is NaN - which would poison the body's
            // temperature for good. The split is therefore computed on a
            // saturated energy and with an explicit zero-ionizing branch.
            float energy = SaturatingProduct(ns.Activity, iso.DecayEnergy) * dtNuclear;
            float fraction = System.Math.Clamp(iso.IonizationFraction, 0.0f, 1.0f);
            float ionizing = fraction <= 0.0f ? 0.0f : energy * fraction;
            scratch.Heat[i] += energy - ionizing;
            scratch.Radiation[i] += ionizing;

            ns.Activity *= DeterministicScalar.Exp(-lambda * dtNuclear);
            if (!float.IsFinite(ns.Activity) || ns.Activity < 0.0f)
            {
                ns.Activity = 0.0f;
            }
            ns.Activity = MinNum(ns.Activity, config.MaxActivity);
        }

        // 2. Fission and the chain reaction.
        if (config.FissionRate > 0.0f)
        {
            StepFission2(bodies, config, scratch, dtNuclear);
        }

        // 3. Deposit the ionizing radiation into the nearby bodies.
        DepositDose2(bodies, config, scratch);
    }

    /// <summary>
    /// Runs the nuclear phase for a 3D world.
    /// </summary>
    /// <param name="bodies">The world bodies.</param>
    /// <param name="config">The nuclear configuration.</param>
    /// <param name="scratch">The per-step scratch.</param>
    /// <param name="dt">The time step, in simulation seconds.</param>
    public static void StepNuclear3(
        List<RigidBody3> bodies,
        NuclearConfig config,
        NuclearScratch scratch,
        float dt)
    {
        // The scratch is cleared *before* the early returns: the world
        // applies the heat budget after this call, and a stale budget
        // would be re-applied every step (an unbounded temperature
        // ramp).
        int n = bodies.Count;
        if (scratch.Heat.Length != n)
        {
            scratch.Heat = new float[n];
            scratch.Radiation = new float[n];
            scratch.Flux = new float[n];
        }
        else
        {
            Array.Clear(scratch.Heat);
            Array.Clear(scratch.Radiation);
            Array.Clear(scratch.Flux);
        }

        if (!config.IsActive() || dt <= 0.0f || bodies.Count == 0)
        {
            return;
        }
        float dtNuclear = dt * config.TimeScale;
        if (dtNuclear <= 0.0f)
        {
            return;
        }

        for (int i = 0; i < n; i++)
        {
            var ns = bodies[i].Nuclear;
            if (ns.Isotope == 0 || ns.Activity <= 0.0f)
            {
                continue;
            }
            var iso = config.GetIsotope(ns.Isotope);
            if (iso == null)
            {
                continue;
            }
            float lambda = iso.DecayConstant();
            if (lambda <= 0.0f)
            {
                ns.Activity = 0.0f;
                continue;
            }
            if (iso.DecayEnergy <= 0.0f)
            {
                continue;
            }
            // The product can overflow to infinity for extreme inputs, and
            // `infinity * 0.0` is NaN - which would poison the body's
            // temperature for good. The split is therefore computed on a
            // saturated energy and with an explicit zero-ionizing branch.
            float energy = SaturatingProduct(ns.Activity, iso.DecayEnergy) * dtNuclear;
            float fraction = System.Math.Clamp(iso.IonizationFraction, 0.0f, 1.0f);
            float ionizing = fraction <= 0.0f ? 0.0f : energy * fraction;
            scratch.Heat[i] += energy - ionizing;
            scratch.Radiation[i] += ionizing;

            ns.Activity *= DeterministicScalar.Exp(-lambda * dtNuclear);
            if (!float.IsFinite(ns.Activity) || ns.Activity < 0.0f)
            {
                ns.Activity = 0.0f;
            }
            ns.Activity = MinNum(ns.Activity, config.MaxActivity);
        }

        // 2. Fission and the chain reaction.
        if (config.FissionRate > 0.0f)
        {
            StepFission3(bodies, config, scratch, dtNuclear);
        }

        // 3. Deposit the ionizing radiation into the nearby bodies.
        DepositDose3(bodies, config, scratch);
    }

    /// <summary>
    /// The fission and chain-reaction step (3D).
    /// </summary>
    private static void StepFission3(
        List<RigidBody3> bodies,
        NuclearConfig config,
        NuclearScratch scratch,
        float dtNuclear)
    {
        float rangeSq = config.NeutronRange * config.NeutronRange;

        // 1. Spread the free neutrons to the neighbours within the range.
        for (int i = 0; i < bodies.Count; i++)
        {
            float available = bodies[i].Nuclear.Neutrons;
            if (available <= 0.0f)
            {
                continue;
            }
            Vector3 origin = bodies[i].Transform.Position;
            int neighbours = 0;
            for (int j = 0; j < bodies.Count; j++)
            {
                if (i == j)
                {
                    continue;
                }
                Vector3 d = origin - bodies[j].Transform.Position;
                if (d.LengthSquared() <= rangeSq)
                {
                    neighbours++;
                }
            }
            if (neighbours == 0)
            {
                continue;
            }
            float share = available / neighbours;
            for (int j = 0; j < bodies.Count; j++)
            {
                if (i == j)
                {
                    continue;
                }
                Vector3 d = origin - bodies[j].Transform.Position;
                if (d.LengthSquared() <= rangeSq)
                {
                    scratch.Flux[j] += share;
                }
            }
        }
        for (int i = 0; i < bodies.Count; i++)
        {
            var ns = bodies[i].Nuclear;
            ns.Neutrons = 0.0f;
        }

        // 2. The delayed neutrons leak out over the delay.
        for (int i = 0; i < bodies.Count; i++)
        {
            var ns = bodies[i].Nuclear;
            if (ns.DelayedNeutrons <= 0.0f)
            {
                continue;
            }
            // A non-positive delay means "release everything now". The
            // explicit branch avoids relying on dt / (the smallest
            // normal) overflowing to -inf.
            float released = config.DelayedNeutronDelay > 0.0f
                ? ns.DelayedNeutrons
                    * (1.0f - DeterministicScalar.Exp(-dtNuclear / config.DelayedNeutronDelay))
                : ns.DelayedNeutrons;
            ns.DelayedNeutrons -= released;
            ns.Neutrons += released;
        }

        // 3. The fissions.
        for (int i = 0; i < bodies.Count; i++)
        {
            var ns = bodies[i].Nuclear;
            if (ns.Isotope == 0 || ns.Fuel <= 0.0f)
            {
                continue;
            }
            var iso = config.GetIsotope(ns.Isotope);
            if (iso == null)
            {
                continue;
            }
            if (!iso.IsFissile || iso.FissionCrossSection <= 0.0f)
            {
                continue;
            }
            float flux = config.AmbientFlux + scratch.Flux[i];
            if (flux <= 0.0f)
            {
                continue;
            }
            float rate = flux * iso.FissionCrossSection * ns.Fuel * config.FissionRate;
            float fissions = rate * dtNuclear;
            if (fissions <= 0.0f)
            {
                continue;
            }
            // The fissions cannot exceed the fuel actually present.
            if (config.FuelPerFission > 0.0f)
            {
                fissions = System.Math.Min(fissions, ns.Fuel / config.FuelPerFission);
            }
            if (fissions <= 0.0f)
            {
                continue;
            }
            ns.Fuel = System.Math.Max(ns.Fuel - fissions * config.FuelPerFission, 0.0f);
            scratch.Heat[i] += SaturatingProduct(fissions, config.FissionEnergy);
            float neutrons = fissions * iso.NeutronYield;
            float delayed = neutrons * System.Math.Clamp(config.DelayedNeutronFraction, 0.0f, 1.0f);
            ns.DelayedNeutrons += delayed;
            ns.Neutrons += neutrons - delayed;
        }

        // 4. Capture (see the 2D counterpart).
        for (int i = 0; i < bodies.Count; i++)
        {
            var ns = bodies[i].Nuclear;
            if (ns.Isotope == 0)
            {
                continue;
            }
            var iso = config.GetIsotope(ns.Isotope);
            if (iso == null || iso.Daughter == 0 || iso.AbsorptionCrossSection <= 0.0f)
            {
                continue;
            }
            float flux = config.AmbientFlux + scratch.Flux[i];
            if (flux <= 0.0f)
            {
                continue;
            }
            float rate = flux * iso.AbsorptionCrossSection * config.ActivationRate;
            if (rate <= 0.0f)
            {
                continue;
            }
            ns.CaptureProgress += rate * dtNuclear;
            if (ns.CaptureProgress >= 1.0f)
            {
                ns.Isotope = iso.Daughter;
                ns.CaptureProgress = 0.0f;
                ns.Activity = 0.0f;
                ns.Fuel = 0.0f;
            }
        }
    }

    /// <summary>The fission and chain-reaction step (2D).</summary>
    private static void StepFission2(
        List<RigidBody2> bodies,
        NuclearConfig config,
        NuclearScratch scratch,
        float dtNuclear)
    {
        float rangeSq = config.NeutronRange * config.NeutronRange;

        // 1. Spread the free neutrons to the neighbours within the range.
        //    Read-only, so the bodies can be borrowed immutably here.
        for (int i = 0; i < bodies.Count; i++)
        {
            float available = bodies[i].Nuclear.Neutrons;
            if (available <= 0.0f)
            {
                continue;
            }
            Vector2 origin = bodies[i].Transform.Position;
            int neighbours = 0;
            for (int j = 0; j < bodies.Count; j++)
            {
                if (i == j)
                {
                    continue;
                }
                Vector2 d = origin - bodies[j].Transform.Position;
                if (d.LengthSquared() <= rangeSq)
                {
                    neighbours++;
                }
            }
            if (neighbours == 0)
            {
                // The neutrons escape.
                continue;
            }
            float share = available / neighbours;
            for (int j = 0; j < bodies.Count; j++)
            {
                if (i == j)
                {
                    continue;
                }
                Vector2 d = origin - bodies[j].Transform.Position;
                if (d.LengthSquared() <= rangeSq)
                {
                    scratch.Flux[j] += share;
                }
            }
        }
        // The neutrons have been handed over (or escaped).
        for (int i = 0; i < bodies.Count; i++)
        {
            var ns = bodies[i].Nuclear;
            ns.Neutrons = 0.0f;
        }

        // 2. The delayed neutrons leak out over the delay and join the
        //    free pool.
        for (int i = 0; i < bodies.Count; i++)
        {
            var ns = bodies[i].Nuclear;
            if (ns.DelayedNeutrons <= 0.0f)
            {
                continue;
            }
            // A non-positive delay means "release everything now". The
            // explicit branch avoids relying on dt / (the smallest
            // normal) overflowing to -inf.
            float released = config.DelayedNeutronDelay > 0.0f
                ? ns.DelayedNeutrons
                    * (1.0f - DeterministicScalar.Exp(-dtNuclear / config.DelayedNeutronDelay))
                : ns.DelayedNeutrons;
            ns.DelayedNeutrons -= released;
            ns.Neutrons += released;
        }

        // 3. The fissions.
        for (int i = 0; i < bodies.Count; i++)
        {
            var ns = bodies[i].Nuclear;
            if (ns.Isotope == 0 || ns.Fuel <= 0.0f)
            {
                continue;
            }
            var iso = config.GetIsotope(ns.Isotope);
            if (iso == null)
            {
                continue;
            }
            if (!iso.IsFissile || iso.FissionCrossSection <= 0.0f)
            {
                continue;
            }

            float flux = config.AmbientFlux + scratch.Flux[i];
            if (flux <= 0.0f)
            {
                continue;
            }
            float rate = flux * iso.FissionCrossSection * ns.Fuel * config.FissionRate;
            float fissions = rate * dtNuclear;
            if (fissions <= 0.0f)
            {
                continue;
            }
            // The fissions cannot exceed the fuel actually present:
            // without this cap a body would keep producing heat and
            // neutrons long after its fuel ran out.
            if (config.FuelPerFission > 0.0f)
            {
                fissions = System.Math.Min(fissions, ns.Fuel / config.FuelPerFission);
            }
            if (fissions <= 0.0f)
            {
                continue;
            }

            ns.Fuel = System.Math.Max(ns.Fuel - fissions * config.FuelPerFission, 0.0f);

            float energy = SaturatingProduct(fissions, config.FissionEnergy);
            scratch.Heat[i] += energy;

            float neutrons = fissions * iso.NeutronYield;
            float delayed = neutrons * System.Math.Clamp(config.DelayedNeutronFraction, 0.0f, 1.0f);
            ns.DelayedNeutrons += delayed;
            ns.Neutrons += neutrons - delayed;
        }

        // 4. Capture: a body with a daughter nuclide absorbs the
        //    neutrons in its flux and becomes that daughter. The capture
        //    is mean-field (a rate, not a per-neutron coin flip), so it
        //    stays deterministic without an RNG.
        for (int i = 0; i < bodies.Count; i++)
        {
            var ns = bodies[i].Nuclear;
            if (ns.Isotope == 0)
            {
                continue;
            }
            var iso = config.GetIsotope(ns.Isotope);
            if (iso == null || iso.Daughter == 0 || iso.AbsorptionCrossSection <= 0.0f)
            {
                continue;
            }
            float flux = config.AmbientFlux + scratch.Flux[i];
            if (flux <= 0.0f)
            {
                continue;
            }
            float rate = flux * iso.AbsorptionCrossSection * config.ActivationRate;
            if (rate <= 0.0f)
            {
                continue;
            }
            ns.CaptureProgress += rate * dtNuclear;
            if (ns.CaptureProgress >= 1.0f)
            {
                ns.Isotope = iso.Daughter;
                ns.CaptureProgress = 0.0f;
                ns.Activity = 0.0f;
                ns.Fuel = 0.0f;
            }
        }
    }

    /// <summary>
    /// Deposits the ionizing radiation into the nearby bodies (2D).
    /// </summary>
    /// <remarks>
    /// The energy is shared out with an inverse-square weight, so the
    /// total deposited energy never exceeds the energy produced: radiation
    /// that reaches no body is simply lost (it left the system).
    /// </remarks>
    private static void DepositDose2(
        List<RigidBody2> bodies,
        NuclearConfig config,
        NuclearScratch scratch)
    {
        float rangeSq = config.DoseRange * config.DoseRange;
        int n = bodies.Count;
        if (scratch.DoseWeights.Length != n)
        {
            scratch.DoseWeights = new float[n];
        }
        else
        {
            Array.Clear(scratch.DoseWeights);
        }

        for (int i = 0; i < n; i++)
        {
            float energy = scratch.RadiationAt(i);
            if (energy <= 0.0f)
            {
                continue;
            }
            Vector2 origin = bodies[i].Transform.Position;
            float total = 0.0f;
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    continue;
                }
                Vector2 d = origin - bodies[j].Transform.Position;
                float d2 = d.LengthSquared();
                if (d2 > rangeSq)
                {
                    continue;
                }
                float w = 1.0f / MaxNum(d2, 1e-6f);
                scratch.DoseWeights[j] = w;
                total += w;
            }
            if (total <= 0.0f)
            {
                // The radiation reached nobody.
                continue;
            }
            float dose = energy * config.DosePerJoule;
            for (int j = 0; j < n; j++)
            {
                if (i == j || scratch.DoseWeights[j] <= 0.0f)
                {
                    continue;
                }
                float share = scratch.DoseWeights[j] / total;
                var ns = bodies[j].Nuclear;
                ns.Dose += dose * share;
                if (!float.IsFinite(ns.Dose) || ns.Dose < 0.0f)
                {
                    ns.Dose = 0.0f;
                }
                ns.Dose = MinNum(ns.Dose, config.MaxDose);
            }
            Array.Clear(scratch.DoseWeights);
        }
    }

    /// <summary>
    /// Deposits the ionizing radiation into the nearby bodies (3D).
    /// </summary>
    private static void DepositDose3(
        List<RigidBody3> bodies,
        NuclearConfig config,
        NuclearScratch scratch)
    {
        float rangeSq = config.DoseRange * config.DoseRange;
        int n = bodies.Count;
        if (scratch.DoseWeights.Length != n)
        {
            scratch.DoseWeights = new float[n];
        }
        else
        {
            Array.Clear(scratch.DoseWeights);
        }

        for (int i = 0; i < n; i++)
        {
            float energy = scratch.RadiationAt(i);
            if (energy <= 0.0f)
            {
                continue;
            }
            Vector3 origin = bodies[i].Transform.Position;
            float total = 0.0f;
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    continue;
                }
                Vector3 d = origin - bodies[j].Transform.Position;
                float d2 = d.LengthSquared();
                if (d2 > rangeSq)
                {
                    continue;
                }
                float w = 1.0f / MaxNum(d2, 1e-6f);
                scratch.DoseWeights[j] = w;
                total += w;
            }
            if (total <= 0.0f)
            {
                continue;
            }
            float dose = energy * config.DosePerJoule;
            for (int j = 0; j < n; j++)
            {
                if (i == j || scratch.DoseWeights[j] <= 0.0f)
                {
                    continue;
                }
                float share = scratch.DoseWeights[j] / total;
                var ns = bodies[j].Nuclear;
                ns.Dose += dose * share;
                if (!float.IsFinite(ns.Dose) || ns.Dose < 0.0f)
                {
                    ns.Dose = 0.0f;
                }
                ns.Dose = MinNum(ns.Dose, config.MaxDose);
            }
            Array.Clear(scratch.DoseWeights);
        }
    }

    /// <summary>
    /// Applies the nuclear heat to the body temperatures (2D).
    /// </summary>
    /// <param name="bodies">The world bodies.</param>
    /// <param name="scratch">The per-step scratch.</param>
    /// <param name="thermalConfig">The thermal configuration (for the clamps).</param>
    public static void ApplyNuclearHeat2(
        List<RigidBody2> bodies,
        NuclearScratch scratch,
        ThermalConfig2 thermalConfig)
    {
        for (int i = 0; i < bodies.Count; i++)
        {
            float heat = scratch.HeatAt(i);
            if (heat <= 0.0f)
            {
                continue;
            }
            var body = bodies[i];
            if (body.IsStatic)
            {
                // Statics are infinite-capacity reservoirs.
                continue;
            }
            float capacity = body.Mass * body.Material.SpecificHeat;
            if (capacity <= 0.0f)
            {
                continue;
            }
            float delta = heat / capacity;
            body.Temperature = System.Math.Clamp(
                body.Temperature + delta,
                thermalConfig.MinTemperature,
                thermalConfig.MaxTemperature);
        }
    }

    /// <summary>
    /// Applies the nuclear heat to the body temperatures (3D).
    /// </summary>
    /// <param name="bodies">The world bodies.</param>
    /// <param name="scratch">The per-step scratch.</param>
    /// <param name="thermalConfig">The thermal configuration (for the clamps).</param>
    public static void ApplyNuclearHeat3(
        List<RigidBody3> bodies,
        NuclearScratch scratch,
        ThermalConfig2 thermalConfig)
    {
        for (int i = 0; i < bodies.Count; i++)
        {
            float heat = scratch.HeatAt(i);
            if (heat <= 0.0f)
            {
                continue;
            }
            var body = bodies[i];
            if (body.IsStatic)
            {
                continue;
            }
            float capacity = body.Mass * body.Material.SpecificHeat;
            if (capacity <= 0.0f)
            {
                continue;
            }
            float delta = heat / capacity;
            body.Temperature = System.Math.Clamp(
                body.Temperature + delta,
                thermalConfig.MinTemperature,
                thermalConfig.MaxTemperature);
        }
    }
}
