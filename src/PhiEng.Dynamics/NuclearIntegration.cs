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
        if (!config.IsActive() || dt <= 0.0f || bodies.Count == 0)
        {
            return;
        }
        float dtNuclear = dt * config.TimeScale;
        if (dtNuclear <= 0.0f)
        {
            return;
        }

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
            float energy = ns.Activity * iso.DecayEnergy * dtNuclear;
            float ionizing = energy * System.Math.Clamp(iso.IonizationFraction, 0.0f, 1.0f);
            scratch.Heat[i] += energy - ionizing;
            scratch.Radiation[i] += ionizing;

            ns.Activity *= DeterministicScalar.Exp(-lambda * dtNuclear);
            if (!float.IsFinite(ns.Activity) || ns.Activity < 0.0f)
            {
                ns.Activity = 0.0f;
            }
            ns.Activity = System.Math.Min(ns.Activity, config.MaxActivity);
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
        if (!config.IsActive() || dt <= 0.0f || bodies.Count == 0)
        {
            return;
        }
        float dtNuclear = dt * config.TimeScale;
        if (dtNuclear <= 0.0f)
        {
            return;
        }

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
            float energy = ns.Activity * iso.DecayEnergy * dtNuclear;
            float ionizing = energy * System.Math.Clamp(iso.IonizationFraction, 0.0f, 1.0f);
            scratch.Heat[i] += energy - ionizing;
            scratch.Radiation[i] += ionizing;

            ns.Activity *= DeterministicScalar.Exp(-lambda * dtNuclear);
            if (!float.IsFinite(ns.Activity) || ns.Activity < 0.0f)
            {
                ns.Activity = 0.0f;
            }
            ns.Activity = System.Math.Min(ns.Activity, config.MaxActivity);
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
            float delay = config.DelayedNeutronDelay > 0.0f
                ? config.DelayedNeutronDelay
                : 1.17549435e-38f;
            float released = ns.DelayedNeutrons * (1.0f - DeterministicScalar.Exp(-dtNuclear / delay));
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
            ns.Fuel = System.Math.Max(ns.Fuel - fissions * config.FuelPerFission, 0.0f);
            scratch.Heat[i] += fissions * config.FissionEnergy;
            float neutrons = fissions * iso.NeutronYield;
            float delayed = neutrons * System.Math.Clamp(config.DelayedNeutronFraction, 0.0f, 1.0f);
            ns.DelayedNeutrons += delayed;
            ns.Neutrons += neutrons - delayed;
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
            float delay = config.DelayedNeutronDelay > 0.0f
                ? config.DelayedNeutronDelay
                : 1.17549435e-38f;
            float released = ns.DelayedNeutrons * (1.0f - DeterministicScalar.Exp(-dtNuclear / delay));
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

            ns.Fuel = System.Math.Max(ns.Fuel - fissions * config.FuelPerFission, 0.0f);

            float energy = fissions * config.FissionEnergy;
            scratch.Heat[i] += energy;

            float neutrons = fissions * iso.NeutronYield;
            float delayed = neutrons * System.Math.Clamp(config.DelayedNeutronFraction, 0.0f, 1.0f);
            ns.DelayedNeutrons += delayed;
            ns.Neutrons += neutrons - delayed;
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
                float w = 1.0f / System.Math.Max(d2, 1e-6f);
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
                ns.Dose = System.Math.Min(ns.Dose, config.MaxDose);
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
                float w = 1.0f / System.Math.Max(d2, 1e-6f);
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
                ns.Dose = System.Math.Min(ns.Dose, config.MaxDose);
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
