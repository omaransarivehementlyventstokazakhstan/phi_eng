using System.Numerics;
using PhiEng.Broadphase;
using PhiEng.Collision;
using PhiEng.Math;

namespace PhiEng.Dynamics;

/// <summary>
/// Cached 3D contact impulses. A struct so entries live inline in their
/// lists (no per-entry heap object). Mirrors Rust <c>CachedContact3D</c>.
/// </summary>
public struct CachedContact3D
{
    /// <summary>Contact point in world space (last frame).</summary>
    public Vector3 Point { get; set; }
    /// <summary>Contact normal (last frame).</summary>
    public Vector3 Normal { get; set; }
    /// <summary>Accumulated normal impulse j_n.</summary>
    public float NormalImpulse { get; set; }
    /// <summary>Accumulated first-tangent impulse.</summary>
    public float Tangent1Impulse { get; set; }
    /// <summary>Accumulated second-tangent impulse.</summary>
    public float Tangent2Impulse { get; set; }
}

/// <summary>
/// 3D physics world: gravity, AABB-tree broadphase, GJK/EPA narrowphase,
/// sequential-impulse solver, warm-start cache. Mirrors Rust <c>World3</c>.
/// </summary>
public class World3
{
    /// <summary>Rigid bodies (ids double as indices).</summary>
    public List<RigidBody3> Bodies { get; set; } = new();
    /// <summary>Broad phase AABB tree.</summary>
    public AABBTree3D Broadphase { get; set; } = new();
    /// <summary>Constraint solver.</summary>
    public Solver3D Solver { get; set; } = new();
    /// <summary>Gravity.</summary>
    public Vector3 Gravity { get; set; } = new Vector3(0, -9.81f, 0);
    /// <summary>
    /// Gravitational force model + integration scheme (see
    /// <see cref="GravityConfig3"/>). Default is the legacy uniform field
    /// with semi-implicit Euler — bit-identical to pre-gravity behavior.
    /// </summary>
    public GravityConfig3 GravityConfig { get; set; } = GravityConfig3.Default;
    /// <summary>
    /// Ambient force fields — spring (SHM), aerodynamic drag, turbulence
    /// (see <see cref="AmbientConfig3"/>). Default is a no-op.
    /// </summary>
    public AmbientConfig3 Ambient { get; set; } = AmbientConfig3.Default;
    /// <summary>
    /// Continuous collision detection via adaptive substepping (see
    /// <see cref="CcdConfig"/>). Default is a no-op (single legacy substep).
    /// </summary>
    public CcdConfig Ccd { get; set; } = CcdConfig.Default;
    /// <summary>Monotonic simulation clock (s), advanced by Dt each step.</summary>
    public float Time { get; set; }
    /// <summary>Next body id.</summary>
    public int NextBodyId { get; set; }
    /// <summary>Fixed timestep.</summary>
    public float Dt { get; set; } = 1.0f / 60.0f;
    /// <summary>Velocity threshold for sleeping.</summary>
    public float LinearSleepThreshold { get; set; } = 0.01f;
    /// <summary>Angular velocity threshold for sleeping.</summary>
    public float AngularSleepThreshold { get; set; } = 0.01f;
    /// <summary>Previous frame's contact impulses, keyed by (bodyA, bodyB).</summary>
    public Dictionary<(int, int), List<CachedContact3D>> ContactCache { get; set; } = new();
    /// <summary>Joint constraints (serial only, solved after contacts).</summary>
    public List<Joint3> Joints { get; set; } = new();
    /// <summary>Next joint id.</summary>
    public int NextJointId { get; set; }

    /// <summary>SPH fluid state (data only; stepping lives in Rust).</summary>
    public FluidState3 Fluid { get; } = new();

    private const float ContactMatchDistSq = 0.04f;

    /// <summary>Runtime validation configuration.</summary>
    public ValidationConfig Validation { get; set; } = ValidationConfig.Default;

    /// <summary>Last validation report.</summary>
    public ValidationReport ValidationReport { get; private set; } = new();

    private readonly Dictionary<int, int> _idToIndex = new();

    // Scratch buffers reused every step so the per-frame narrowphase
    // performs no allocations in the steady state.
    private readonly List<ContactConstraint3D> _constraints = new();

    /// <summary>
    /// Thermal simulation tunables for this world. The thermal phase
    /// runs after velocity integration and before the sleep check, exactly
    /// as in the Rust core. Disabled by default, so default trajectories
    /// are bit-identical.
    /// </summary>
    public ThermalConfig2 Thermal
    {
        get => _thermal.Config;
        set => _thermal.Config = value;
    }

    /// <summary>Per-world thermal scratch (heat flux, cached areas).</summary>
    private readonly ThermalState2 _thermal = new();

    /// <summary>
    /// The nuclear simulation state (see <see cref="NuclearConfig"/>).
    /// Disabled by default.
    /// </summary>
    public NuclearState2 Nuclear => _nuclear;

    private readonly NuclearState2 _nuclear = new();
    private readonly List<(int, int)> _pairs = new();
    private readonly List<Vector3> _points = new();
    private readonly List<Vector3> _clipTmp = new();
    private readonly GjkEpa3D.EpaScratch _epaScratch = new();
    private readonly HashSet<(int, int)> _liveKeys = new();
    private readonly List<(int, int)> _staleKeys = new();
    private readonly List<bool> _contactTouch = new();
    /// <summary>The contact events of the last step.</summary>
    private readonly List<ContactEvent3> _contactEvents = new();
    /// <summary>The sensor pairs touching this step, with their contacts.</summary>
    private List<(int, int, Vector3, Vector3)> _sensorTouching = new();
    /// <summary>The sensor pairs that were touching last step, with their contacts.</summary>
    private List<(int, int, Vector3, Vector3)> _lastSensorTouching = new();
    /// <summary>The pairs that already reported a *began* event this step.</summary>
    private readonly HashSet<(uint, uint)> _announcedPairs = new();
    /// <summary>The sensor pairs that ended this step, with their last contact.</summary>
    private readonly List<(int, int, Vector3, Vector3)> _sensorEnded = new();
    /// <summary>The last known contact of every pair, so an *ended* event can say where the pair was.</summary>
    private readonly Dictionary<(int, int), (Vector3, Vector3)> _lastContacts = new();
    /// <summary>The contacts a sensor pair reported.</summary>
    private readonly List<ContactConstraint3D> _sensorContacts = new();

    /// <summary>Adds a rigid body (validates shape and density). Returns the body id.</summary>
    public int AddBody(RigidBody3 body)
    {
        if (body == null) throw new ArgumentNullException(nameof(body));
        if (body.Material == null) throw new ArgumentNullException(nameof(body), "material must be set");
        ShapeValidation.ValidatePosition(body.Transform.Position);
        if (body.Shape != null)
        {
            ShapeValidation.ValidateShape3(body.Shape);
            ShapeValidation.ValidateDensity(body.IsStatic, body.Material.Density);
        }
        int id = NextBodyId;
        body.Id = id;
        NextBodyId++;
        _idToIndex[id] = Bodies.Count;
        Bodies.Add(body);
        return id;
    }

    /// <summary>Adds an immovable body.</summary>
    public int AddStaticBody(Vector3 position, Shape3 shape)
    {
        var body = RigidBody3.StaticBody(NextBodyId);
        var t = body.Transform; t.Position = position; body.Transform = t;
        body.SetShape(shape, 1.0f);
        return AddBody(body);
    }

    /// <summary>Adds a dynamic body with mass derived from density.</summary>
    public int AddDynamicBody(Vector3 position, Shape3 shape, float density)
    {
        var body = new RigidBody3(NextBodyId);
        var t = body.Transform; t.Position = position; body.Transform = t;
        body.SetShape(shape, density);
        return AddBody(body);
    }

    /// <summary>
    /// Add a 3D joint to the world. Ids double as list indices; the joint
    /// must carry validated endpoints. Mirrors the Rust <c>add_joint</c>.
    /// </summary>
    public int AddJoint(Joint3 joint)
    {
        if (joint == null) throw new ArgumentNullException(nameof(joint));
        var (a, b) = joint.Bodies();
        if (a >= Bodies.Count || b >= Bodies.Count)
            throw new ArgumentException("unknown or stale body id");
        if (a == b)
            throw new ArgumentException("joint requires two distinct bodies");
        int id = NextJointId;
        joint.SetId(id);
        NextJointId++;
        Joints.Add(joint);
        return id;
    }

    /// <summary>
    /// Remove a 3D joint by id. Ids double as list indices; removal shifts
    /// the tail and reassigns ids. Mirrors the Rust <c>remove_joint</c>.
    /// </summary>
    public void RemoveJoint(int jointId)
    {
        if (jointId < 0 || jointId >= Joints.Count) return;
        Joints.RemoveAt(jointId);
        for (int i = 0; i < Joints.Count; i++)
        {
            var j = Joints[i];
            j.SetId(i);
            Joints[i] = j;
        }
        NextJointId = Joints.Count;
    }

    /// <summary>
    /// Remove a body. Ids double as list indices, so removal shifts the
    /// tail and reassigns ids; contact-cache keys are remapped through the
    /// shift (entries touching the removed body are dropped, survivors keep
    /// warm-start impulses), re-canonicalized to (min, max). Joints touching
    /// the removed body are dropped and survivors shift. Mirrors the Rust
    /// <c>remove_body</c>.
    /// </summary>
    public void RemoveBody(int bodyId)
    {
        if (bodyId < 0 || bodyId >= Bodies.Count) return;
        int last = Bodies.Count - 1;
        Joints.RemoveAll(j =>
        {
            var (a, b) = j.Bodies();
            return a == bodyId || b == bodyId;
        });
        Bodies.RemoveAt(bodyId);
        _idToIndex.Clear();
        for (int i = 0; i < Bodies.Count; i++)
        {
            Bodies[i].Id = i;
            _idToIndex[i] = i;
        }
        NextBodyId = Bodies.Count;
        for (int i = 0; i < Joints.Count; i++)
        {
            var j = Joints[i];
            j.Shift(bodyId);
            Joints[i] = j;
        }
        // The pairs that touched the removed body are gone: their
        // contacts ended with the body.
        var dead = new List<(int, int)>();
        foreach (var key in ContactCache.Keys)
            if (key.Item1 == bodyId || key.Item2 == bodyId) dead.Add(key);
        dead.Sort();
        if (CollectContactEvents)
        {
            foreach (var key in dead)
            {
                var pair = EventPair(key.Item1, key.Item2);
                var lastContact = _lastContacts.TryGetValue(key, out var lc)
                    ? lc
                    : (Vector3.Zero, Vector3.Zero);
                _contactEvents.Add(new ContactEvent3(
                    pair.Item1, pair.Item2,
                    ContactEventKind.Ended, lastContact.Item1, lastContact.Item2));
            }
        }

        // The body list shifts down above the gap, so every survivor
        // above it drops by one. The pair keys are remapped through the
        // same shift, and the pairs that touched the removed body are
        // dropped: those contacts are gone.
        var old = new List<((int, int), List<CachedContact3D>)>();
        foreach (var kv in ContactCache) old.Add((kv.Key, kv.Value));
        ContactCache.Clear();
        foreach (var ((a, b), entries) in old)
        {
            if (a == bodyId || b == bodyId) continue;
            int na = a > bodyId ? a - 1 : a;
            int nb = b > bodyId ? b - 1 : b;
            var key = na < nb ? (na, nb) : (nb, na);
            if (!ContactCache.TryGetValue(key, out var list) || list == null)
            {
                list = new List<CachedContact3D>();
                ContactCache[key] = list;
            }
            list.AddRange(entries);
        }
    }

    /// <summary>
    /// Steps the simulation: gravity, broadphase, narrowphase, solve,
    /// integrate, sleep. With CCD enabled (see <see cref="CcdConfig"/>),
    /// the step splits into substeps so no body travels more than
    /// MaxMovement per substep; disabled (default) is a single legacy
    /// substep, bit-identical to pre-CCD behavior.
    /// </summary>
    /// <summary>
    /// The bodies whose AABB overlaps the query box.
    /// </summary>
    /// <remarks>
    /// The ids come back in ascending order, and a body is reported only
    /// when the filter accepts it. The scan is over the *current* body
    /// positions, not over the broad phase's grid, which holds the
    /// positions from before the last solve.
    /// </remarks>
    public List<uint> OverlapAabb(AABB3 aabb, QueryFilter filter)
    {
        var result = new List<uint>();
        foreach (var body in Bodies)
        {
            if (!filter.Accepts(body)) continue;
            var (min, max) = GetBodyAABB(body);
            if (min.X <= aabb.Max.X && max.X >= aabb.Min.X
                && min.Y <= aabb.Max.Y && max.Y >= aabb.Min.Y
                && min.Z <= aabb.Max.Z && max.Z >= aabb.Min.Z)
            {
                result.Add((uint)body.Id);
            }
        }
        return result;
    }

    /// <summary>
    /// Collect the contact events (a pair began or stopped touching).
    ///
    /// Off by default: the events cost a pass over the contact cache,
    /// and a scene that never reads them should not pay for it. When
    /// on, the events of the last step are in
    /// <see cref="ContactEvents"/>.
    /// </summary>
    public bool CollectContactEvents { get; set; }

    /// <summary>
    /// Cast a ray into the world and return the first thing it hits.
    /// </summary>
    /// <remarks>
    /// See <see cref="World2.Raycast"/>. The convex shapes (a convex
    /// hull and a convex mesh) are not tested yet: a ray that would hit
    /// one is reported as a miss.
    /// </remarks>
    public (uint Body, float T, Vector3 Point, Vector3 Normal)? Raycast(
        Vector3 origin, Vector3 direction, float maxT, QueryFilter filter)
    {
        (uint Body, float T, Vector3 Point, Vector3 Normal)? best = null;
        foreach (var body in Bodies)
        {
            if (!filter.Accepts(body)) continue;
            // A sensor is a trigger volume, not a solid: a ray passes
            // through it.
            if (body.IsSensor) continue;
            if (body.Shape == null) continue;
            Vector3 point;
            Vector3 normal;
            float t;
            bool hit;
            switch (body.Shape)
            {
                case SphereShape3 sphere:
                    hit = Collision.Query.RaySphere(
                        origin, direction, body.Transform.Position, sphere.Radius,
                        out t, out point, out normal);
                    break;
                case BoxShape3 box:
                    hit = Collision.Query.RayObb(
                        origin, direction, body.Transform.Position, body.Transform.Rotation,
                        box.HalfExtents, out t, out point, out normal);
                    break;
                case CapsuleShape3 capsule:
                    // The capsule axis is the body's local Y.
                    var axis = Vector3.Transform(Vector3.UnitY, body.Transform.Rotation)
                        * capsule.HalfLength;
                    hit = Collision.Query.RayCapsule(
                        origin, direction,
                        body.Transform.Position + axis, body.Transform.Position - axis,
                        capsule.Radius, out t, out point, out normal);
                    break;
                default:
                    hit = false;
                    t = 0.0f;
                    point = Vector3.Zero;
                    normal = Vector3.Zero;
                    break;
            }
            if (!hit || t > maxT) continue;
            if (best == null || t < best.Value.T)
            {
                best = ((uint)body.Id, t, point, normal);
            }
        }
        return best;
    }

    /// <summary>The contact events of the last step.</summary>
    /// <remarks>
    /// One event per pair that began or stopped touching. The events
    /// are only collected when <see cref="CollectContactEvents"/> is
    /// on, and they are cleared at the start of the next step.
    /// </remarks>
    public IReadOnlyList<ContactEvent3> ContactEvents => _contactEvents;

    public void Step(float dt)
    {
        // Reset validation report once for the whole step; substeps merge.
        ValidationReport = new ValidationReport();

        uint n = Ccd.IsActive() ? Ccd.Substeps(dt, CcdPeakSpeed()) : 1u;
        if (n <= 1)
        {
            StepInner(dt);
        }
        else
        {
            float sub = dt / n;
            for (uint s = 0; s < n; s++)
            {
                StepInner(sub);
            }
        }

        // 8. Advance the simulation clock (drives time-varying ambient fields).
        Time += dt;
    }

    /// <summary>
    /// Peak effective body speed (linear plus angular sweep at the body
    /// extents) for CCD substepping. Static bodies carry zero velocity,
    /// so they contribute nothing.
    /// </summary>
    private float CcdPeakSpeed()
    {
        float peak = 0.0f;
        foreach (var body in Bodies)
        {
            if (body.IsStatic)
            {
                continue;
            }
            var (min, max) = GetBodyAABB(body);
            float extent = (max - min).Length() * 0.5f;
            float eff = body.Velocity.Length() + body.AngularVelocity.Length() * extent;
            peak = System.Math.Max(peak, eff);
        }
        return peak;
    }

    private void StepInner(float dt)
    {
        // 1. Gravity kick (see GravityIntegration): uniform background
        // field plus Central/N-body terms per GravityConfig, one
        // full-step kick (Euler) or the first half-kick (Verlet).
        bool verlet = GravityConfig.Integrator == Integrator.VelocityVerlet;
        float h = verlet ? dt * 0.5f : dt;
        GravityIntegration.Kick3(Bodies, GravityConfig, Gravity, h, verlet);

        // Ambient forces (spring/SHM, drag, turbulence), same slot/split
        // as the gravity kick above.
        if (Ambient.IsActive())
        {
            AmbientIntegration.Kick3(Bodies, Ambient, Time, h);
        }

        // Fluid phase - SPH particles and two-way body coupling (no-op
        // when the fluid holds no particles).
        FluidIntegration3.StepFluid3(Fluid, Bodies, Gravity, Ambient, Time, dt);

        UpdateBroadphase();
        _pairs.Clear();
        Broadphase.CollectPairs(_pairs);

        // The events of the last step are the caller's to read; they are
        // cleared when the next step starts.
        _contactEvents.Clear();

        _constraints.Clear();
        var constraints = _constraints;
        _points.Clear();
        _clipTmp.Clear();
        _sensorContacts.Clear();
        foreach (var (a, b) in _pairs)
        {
            // A sensor reports contacts but never pushes: its contacts
            // go to the sensor list, not to the solver.
            if (Bodies[a].IsSensor || Bodies[b].IsSensor)
            {
                TestCollisionsInto(a, b, _sensorContacts, _points, _clipTmp, _epaScratch);
            }
            else
            {
                TestCollisionsInto(a, b, constraints, _points, _clipTmp, _epaScratch);
            }
        }

        // 3a. A sensor's contacts are events, not contacts: the pair set
        // is compared with last step's, and a pair that is new (or gone)
        // produces an event.
        // The state is kept whether or not the events are collected: a
        // pair that was already touching when the collection was switched
        // on must not be reported as a new touch.
        if (_sensorContacts.Count > 0 || _lastSensorTouching.Count > 0)
        {
            SensorEvents();
        }

        if (_contactTouch.Count != Bodies.Count)
        {
            _contactTouch.Clear();
            for (int ti = 0; ti < Bodies.Count; ti++) _contactTouch.Add(false);
        }
        for (int ti = 0; ti < _contactTouch.Count; ti++) _contactTouch[ti] = false;
        for (int ci = 0; ci < constraints.Count; ci++)
        {
            var c = constraints[ci];
            _contactTouch[c.BodyA] = true;
            _contactTouch[c.BodyB] = true;
            bool isNew = !ContactCache.ContainsKey((c.BodyA, c.BodyB));
            // A pair that was not touching last step began touching this
            // step: that is the event a trigger wants.
            if (isNew && CollectContactEvents)
            {
                var pair = EventPair(c.BodyA, c.BodyB);
                // One event per *pair*: a pair that touches at two points
                // began touching once.
                if (_announcedPairs.Add((pair.Item1, pair.Item2)))
                {
                    _contactEvents.Add(new ContactEvent3(
                        pair.Item1, pair.Item2,
                        ContactEventKind.Began, c.Point, c.Normal));
                }
            }
            if (isNew && (Bodies[c.BodyA].IsSleeping || Bodies[c.BodyB].IsSleeping))
            {
                WakeBody(c.BodyA);
                WakeBody(c.BodyB);
            }
            if (!ContactCache.TryGetValue((c.BodyA, c.BodyB), out var entries)) continue;
            CachedContact3D? best = null;
            float bestDist = ContactMatchDistSq;
            foreach (var e in entries)
            {
                if (Vector3.Dot(e.Normal, c.Normal) <= 0) continue;
                float d = (e.Point - c.Point).LengthSquared();
                if (d < bestDist) { bestDist = d; best = e; }
            }
            if (best != null)
            {
                var match = best.Value;
                c.NormalImpulse = match.NormalImpulse;
                c.Tangent1Impulse = match.Tangent1Impulse;
                c.Tangent2Impulse = match.Tangent2Impulse;
                constraints[ci] = c;
            }
        }

        // Validate constraints before solving
        var constraintReport = Validation3D.ValidateConstraints(Bodies, constraints, Validation);
        ValidationReport.Merge(constraintReport);

        // Body ids double as list indices (see AddBody/RemoveBody), so
        // constraints address Bodies directly with no remapping.
        Solver.Solve(Bodies, constraints, dt);

        // 4b. Solve joints (after contacts, same iteration budget).
        // Joints own their impulses, so warm starting is direct.
        // Skipped entirely when the world holds no joints (zero cost).
        // Frozen joints are skipped; a sleeping end with an awake
        // partner wakes first (see JointSleepGate).
        if (Joints.Count > 0)
        {
            foreach (var joint in Joints)
                if (!JointSleepGate(joint)) joint.WarmStart(Bodies);
            for (int it = 0; it < Solver.Iterations; it++)
                foreach (var joint in Joints)
                    if (!JointSleepGate(joint)) joint.Solve(Bodies, dt);
        }

        // Remember each pair's last known contact before the lists are
        // cleared, so an *ended* event can still say where the pair was.
        if (CollectContactEvents)
        {
            _lastContacts.Clear();
            foreach (var entry in ContactCache)
            {
                if (entry.Value.Count > 0)
                {
                    var first = entry.Value[0];
                    _lastContacts[entry.Key] = (first.Point, first.Normal);
                }
            }
        }
        foreach (var list in ContactCache.Values) list.Clear();
        _liveKeys.Clear();
        foreach (var c in constraints)
        {
            var key = (c.BodyA, c.BodyB);
            _liveKeys.Add(key);
            if (!ContactCache.TryGetValue(key, out var list) || list == null)
            {
                list = new List<CachedContact3D>(4);
                ContactCache[key] = list;
            }
            list.Add(new CachedContact3D
            {
                Point = c.Point, Normal = c.Normal,
                NormalImpulse = c.NormalImpulse,
                Tangent1Impulse = c.Tangent1Impulse,
                Tangent2Impulse = c.Tangent2Impulse,
            });
        }
        if (_liveKeys.Count != ContactCache.Count)
        {
            _staleKeys.Clear();
            foreach (var key in ContactCache.Keys)
                if (!_liveKeys.Contains(key)) _staleKeys.Add(key);
            // The pairs are sorted, so the events come out in the same
            // order however the cache was walked.
            _staleKeys.Sort();
            foreach (var key in _staleKeys)
            {
                // A pair that was touching last step and is not touching
                // now stopped touching: that is the other half of the
                // event. The contact data is the pair's last known one.
                if (CollectContactEvents
                    && !(key.Item1 < Bodies.Count && key.Item2 < Bodies.Count
                         && (Bodies[key.Item1].IsSensor || Bodies[key.Item2].IsSensor)))
                {
                    var pair = EventPair(key.Item1, key.Item2);
                    var lastContact = _lastContacts.TryGetValue(key, out var lc)
                        ? lc
                        : (Vector3.Zero, Vector3.Zero);
                    _contactEvents.Add(new ContactEvent3(
                        pair.Item1, pair.Item2,
                        ContactEventKind.Ended, lastContact.Item1, lastContact.Item2));
                }
                // A vanished pair is lost contact: wake any sleeping party.
                if ((key.Item1 < Bodies.Count && Bodies[key.Item1].IsSleeping)
                    || (key.Item2 < Bodies.Count && Bodies[key.Item2].IsSleeping))
                {
                    WakeBody(key.Item1);
                    WakeBody(key.Item2);
                }
                ContactCache.Remove(key);
            }
        }

        if (verlet)
        {
            // 5v. Verlet drift + second half-kick (forces recomputed at
            // the new positions). The contact solver above ran unchanged.
            // Sleeping bodies are skipped: frozen until woken.
            foreach (var body in Bodies)
                if (!body.IsSleeping) body.IntegrateDrift(dt);
            GravityIntegration.Kick3(Bodies, GravityConfig, Gravity, dt * 0.5f, true);
            if (Ambient.IsActive())
            {
                AmbientIntegration.Kick3(Bodies, Ambient, Time, dt * 0.5f);
            }
        }
        else
        {
            // Sleeping bodies are frozen until woken.
            foreach (var body in Bodies)
                if (!body.IsSleeping) body.Integrate(dt);
        }

        // 5c. Nuclear decay and fission (see World2.StepInner).
        if (_nuclear.Config.IsActive())
        {
            NuclearIntegration.StepNuclear3(Bodies, _nuclear.Config, _nuclear.Scratch, dt);
            NuclearIntegration.ApplyNuclearHeat3(Bodies, _nuclear.Scratch, _thermal.Config);
        }

        // 5d. Thermal transfer (conduction / radiation / convection),
        // then 5e. Thermal expansion. The phase runs after integration
        // and before the sleep check, matching the Rust core. It does not
        // modify velocity, so it neither wakes sleeping bodies nor is
        // affected by sleep.
        ThermalIntegration.StepThermal3(Bodies, constraints, _thermal.Config, _thermal, dt);
        ThermalIntegration.ApplyExpansion3(Bodies, _thermal.Config);

        // Validate bodies + update sleep in a single fused sweep
        // (bit-identical to validate-all-then-sleep-all; mirrors Rust).
        // Sleep needs slowness AND support (a current contact): ballistic
        // apexes and drifting bodies never sleep. Sleeping bodies have
        // zeroed velocities every step.
        float linSleep = LinearSleepThreshold;
        float angSleep = AngularSleepThreshold;
        for (int i = 0; i < Bodies.Count; i++)
        {
            var bodyReport = Validation3D.ValidateBody(Bodies[i], Validation);
            ValidationReport.Merge(bodyReport);
            var body = Bodies[i];
            if (body.IsStatic) continue;
            if (body.IsSleeping)
            {
                body.Velocity = Vector3.Zero;
                body.AngularVelocity = Vector3.Zero;
                continue;
            }
            if (body.ShouldSleep(linSleep, angSleep) && _contactTouch[i])
            {
                body.IsSleeping = true;
                body.Velocity = Vector3.Zero;
                body.AngularVelocity = Vector3.Zero;
            }
        }
    }

    private void UpdateBroadphase()
    {
        Broadphase.Clear();
        foreach (var body in Bodies)
        {
            var (min, max) = GetBodyAABB(body);
            Broadphase.Insert(body.Id, min, max);
        }
    }

    private static (Vector3, Vector3) GetBodyAABB(RigidBody3 body)
    {
        if (body.Shape == null) return (body.Transform.Position, body.Transform.Position);
        switch (body.Shape)
        {
            case SphereShape3 s:
                var r = new Vector3(s.Radius, s.Radius, s.Radius);
                return (body.Transform.Position - r, body.Transform.Position + r);
            case BoxShape3 b:
            {
                // Exact AABB from the 8 rotated corners (no allocation).
                var he = b.HalfExtents;
                var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
                var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
                for (int k = 0; k < 8; k++)
                {
                    var local = new Vector3(
                        ((k & 1) == 0 ? -he.X : he.X),
                        ((k & 2) == 0 ? -he.Y : he.Y),
                        ((k & 4) == 0 ? -he.Z : he.Z));
                    var w = body.Transform.TransformPoint(local);
                    min = Vector3.Min(min, w);
                    max = Vector3.Max(max, w);
                }
                return (min, max);
            }
            case CapsuleShape3 c:
            {
                // Segment endpoints in world space, padded by radius
                // (rotation-aware; the old axis-aligned version missed
                // collisions for rotated capsules).
                var axis = Vector3.Transform(new Vector3(0, c.HalfLength, 0), body.Transform.Rotation);
                var pa = body.Transform.Position + axis;
                var pb = body.Transform.Position - axis;
                var rr = new Vector3(c.Radius, c.Radius, c.Radius);
                return (Vector3.Min(pa, pb) - rr, Vector3.Max(pa, pb) + rr);
            }
            case ConvexHullShape3 h:
            {
                if (h.Vertices.Count == 0) return (body.Transform.Position, body.Transform.Position);
                var min = body.Transform.TransformPoint(h.Vertices[0]);
                var max = min;
                for (int k = 1; k < h.Vertices.Count; k++)
                {
                    var w = body.Transform.TransformPoint(h.Vertices[k]);
                    min = Vector3.Min(min, w);
                    max = Vector3.Max(max, w);
                }
                return (min, max);
            }
            default:
                return (body.Transform.Position, body.Transform.Position);
        }
    }

    /// <summary>
    /// Work out which sensor pairs began and which ended.
    /// </summary>
    private void SensorEvents()
    {
        bool collect = CollectContactEvents;

        // The pairs touching a sensor now, in order and without repeats,
        // with the contact each of them reported.
        _sensorTouching.Clear();
        foreach (var c in _sensorContacts)
        {
            var key = (c.BodyA, c.BodyB);
            if (_sensorTouching.Count == 0
                || _sensorTouching[^1].Item1 != key.Item1
                || _sensorTouching[^1].Item2 != key.Item2)
            {
                _sensorTouching.Add((c.BodyA, c.BodyB, c.Point, c.Normal));
            }
        }

        // Merge the two sorted lists and report what changed.
        int i = 0, j = 0;
        _sensorEnded.Clear();
        while (i < _sensorTouching.Count || j < _lastSensorTouching.Count)
        {
            bool hasNow = i < _sensorTouching.Count;
            bool hadBefore = j < _lastSensorTouching.Count;
            var now = hasNow ? (_sensorTouching[i].Item1, _sensorTouching[i].Item2) : (0, 0);
            var before = hadBefore ? (_lastSensorTouching[j].Item1, _lastSensorTouching[j].Item2) : (0, 0);
            if (hasNow && hadBefore && now == before)
            {
                i++;
                j++;
            }
            else if (hasNow && (!hadBefore || PairLess(now, before)))
            {
                if (collect)
                {
                    var entry = _sensorTouching[i];
                    var pair = EventPair(now.Item1, now.Item2);
                    _contactEvents.Add(new ContactEvent3(
                        pair.Item1, pair.Item2, ContactEventKind.Began,
                        entry.Item3, entry.Item4));
                }
                i++;
            }
            else if (hadBefore)
            {
                // The pair ended: it keeps the last contact it had, so
                // the event can say where it was.
                _sensorEnded.Add(_lastSensorTouching[j]);
                j++;
            }
            else
            {
                break;
            }
        }

        // The pairs that ended are reported after the ones that began.
        for (int k = 0; k < _sensorEnded.Count; k++)
        {
            if (!collect) break;
            var entry = _sensorEnded[k];
            var pair = EventPair(entry.Item1, entry.Item2);
            _contactEvents.Add(new ContactEvent3(
                pair.Item1, pair.Item2, ContactEventKind.Ended,
                entry.Item3, entry.Item4));
        }

        // This step's sensor pairs are next step's.
        var swap = _sensorTouching;
        _sensorTouching = _lastSensorTouching;
        _lastSensorTouching = swap;
    }

    /// <summary>The two bodies of a pair, in a canonical order.</summary>
    /// <remarks>
    /// The narrow phase does not promise an order, so the events put
    /// the smaller id first: an event says <em>which two</em> bodies
    /// began or stopped touching, and that must not depend on the
    /// order the narrow phase happened to produce.
    /// </remarks>
    private static (uint, uint) EventPair(int a, int b)
        => a <= b ? ((uint)a, (uint)b) : ((uint)b, (uint)a);

    /// <summary>Whether a pair sorts before another (the pairs are in
    /// sorted order, so the two lists merge).</summary>
    private static bool PairLess((int, int) a, (int, int) b)
        => a.Item1 < b.Item1 || (a.Item1 == b.Item1 && a.Item2 < b.Item2);

    /// <summary>The contact point of a sensor pair, if it has one.</summary>
    private Vector3 SensorPoint((int, int) key)
    {
        foreach (var c in _sensorContacts)
            if (c.BodyA == key.Item1 && c.BodyB == key.Item2) return c.Point;
        return Vector3.Zero;
    }

    /// <summary>The contact normal of a sensor pair, if it has one.</summary>
    private Vector3 SensorNormal((int, int) key)
    {
        foreach (var c in _sensorContacts)
            if (c.BodyA == key.Item1 && c.BodyB == key.Item2) return c.Normal;
        return Vector3.Zero;
    }

    private void TestCollisionsInto(
        int a, int b,
        List<ContactConstraint3D> output,
        List<Vector3> points, List<Vector3> clipTmp,
        GjkEpa3D.EpaScratch epaScratch)
    {
        if (!_idToIndex.TryGetValue(a, out int ia) || !_idToIndex.TryGetValue(b, out int ib))
            return;
        var bodyA = Bodies[ia];
        var bodyB = Bodies[ib];
        // Static-static pairs can never produce an impulse; skip the
        // narrowphase. Bit-identical (their constraints would solve to zero).
        if (bodyA.IsStatic && bodyB.IsStatic) return;
        // Collision layers: each body must accept the other's
        // layer. The default filter accepts everything, so an
        // unconfigured scene is unaffected.
        if (!CollisionFilter.Allows(bodyA, bodyB)) return;
        float friction = PhysicsMaterial.CombineFriction(bodyA.Material.Friction, bodyB.Material.Friction);
        float restitution = PhysicsMaterial.CombineRestitution(bodyA.Material.Restitution, bodyB.Material.Restitution);

        void Build(List<Vector3> pts, Vector3 normal, float depth)
        {
            int count = System.Math.Min(4, pts.Count);
            for (int i = 0; i < count; i++)
                output.Add(new ContactConstraint3D(
                    a, b, pts[i], normal, depth, friction, restitution));
        }

        // Box-box: exact 3D SAT (stable face normals).
        if (bodyA.Shape is BoxShape3 boxA && bodyB.Shape is BoxShape3 boxB)
        {
            var sat = Sat3D.ObbObb(
                bodyA.Transform.Position, boxA.HalfExtents, bodyA.Transform.Rotation,
                bodyB.Transform.Position, boxB.HalfExtents, bodyB.Transform.Rotation);
            if (sat == null) return;
            var (normal, depth, kind) = sat.Value;
            depth = MathF.Max(0, depth);
            points.Clear();
            if (kind == ObbContactKind.FaceA)
            {
                Contact3D.BoxBoxContactPointsInto(
                    bodyA.Transform.Position, boxA.HalfExtents, bodyA.Transform.Rotation,
                    bodyB.Transform.Position, boxB.HalfExtents, bodyB.Transform.Rotation,
                    normal, points, clipTmp);
                if (points.Count == 0)
                    Contact3D.BoxBoxContactPointsInto(
                        bodyB.Transform.Position, boxB.HalfExtents, bodyB.Transform.Rotation,
                        bodyA.Transform.Position, boxA.HalfExtents, bodyA.Transform.Rotation,
                        -normal, points, clipTmp);
            }
            else if (kind == ObbContactKind.FaceB)
            {
                Contact3D.BoxBoxContactPointsInto(
                    bodyB.Transform.Position, boxB.HalfExtents, bodyB.Transform.Rotation,
                    bodyA.Transform.Position, boxA.HalfExtents, bodyA.Transform.Rotation,
                    -normal, points, clipTmp);
            }
            else
            {
                var supA = MakeSupport(bodyA);
                var supB = MakeSupport(bodyB);
                points.Add((supA(normal) + supB(-normal)) * 0.5f);
            }
            if (points.Count == 0) return;
            Build(points, normal, depth);
            return;
        }

        // Sphere-sphere: analytic.
        if (bodyA.Shape is SphereShape3 spA && bodyB.Shape is SphereShape3 spB)
        {
            var delta = bodyB.Transform.Position - bodyA.Transform.Position;
            float dist = delta.Length();
            float depth = spA.Radius + spB.Radius - dist;
            if (depth < 0) return;
            var normal = dist > 1e-6f ? delta / dist : Vector3.UnitX;
            output.Add(new ContactConstraint3D(
                a, b, bodyA.Transform.Position + normal * spA.Radius, normal, depth, friction, restitution));
            return;
        }

        // Sphere-box and box-sphere: closest-point normal (exact).
        if (bodyA.Shape is SphereShape3 sA && bodyB.Shape is BoxShape3 bB)
        {
            var (q, _) = Contact3D.ClosestPointOnBox(
                bodyA.Transform.Position, bodyB.Transform.Position, bB.HalfExtents, bodyB.Transform.Rotation);
            var diff = bodyA.Transform.Position - q;
            float dist = diff.Length();
            if (dist > sA.Radius) return;
            Vector3 normal = dist > 1e-6f ? -diff / dist : FallbackNormal(bodyA, bodyB, epaScratch);
            if (normal.LengthSquared() < 0.5f) return;
            output.Add(new ContactConstraint3D(
                a, b, q, normal, MathF.Max(0, sA.Radius - dist), friction, restitution));
            return;
        }
        if (bodyA.Shape is BoxShape3 bA && bodyB.Shape is SphereShape3 sB)
        {
            var (q, _) = Contact3D.ClosestPointOnBox(
                bodyB.Transform.Position, bodyA.Transform.Position, bA.HalfExtents, bodyA.Transform.Rotation);
            var diff = bodyB.Transform.Position - q;
            float dist = diff.Length();
            if (dist > sB.Radius) return;
            Vector3 normal = dist > 1e-6f ? diff / dist : FallbackNormal(bodyA, bodyB, epaScratch);
            if (normal.LengthSquared() < 0.5f) return;
            output.Add(new ContactConstraint3D(
                a, b, q, normal, MathF.Max(0, sB.Radius - dist), friction, restitution));
            return;
        }

        // Capsules / hulls: GJK/EPA. Witness refinement: a box support is
        // a far corner, so a raw support midpoint can sit meters from the
        // contact — the lever arm collapses the effective mass and the
        // solver pushes ~nothing. Any box side is replaced by the closest
        // box-surface point to the other side's witness. Mirrors Rust.
        var supA2 = MakeSupport(bodyA);
        var supB2 = MakeSupport(bodyB);
        var hit = GjkEpa3D.GjkEpa(supA2, supB2, epaScratch);
        if (hit == null) return;
        var pa = supA2(hit.Normal);
        var pb = supB2(-hit.Normal);
        var wa = bodyA.Shape is BoxShape3 ba
            ? Contact3D.ClosestPointOnBox(pb, bodyA.Transform.Position, ba.HalfExtents, bodyA.Transform.Rotation).Point
            : pa;
        var wb = bodyB.Shape is BoxShape3 bb
            ? Contact3D.ClosestPointOnBox(pa, bodyB.Transform.Position, bb.HalfExtents, bodyB.Transform.Rotation).Point
            : pb;
        output.Add(new ContactConstraint3D(
            a, b, (wa + wb) * 0.5f,
            hit.Normal, MathF.Max(0, hit.Depth), friction, restitution));
    }

    private static Vector3 FallbackNormal(RigidBody3 bodyA, RigidBody3 bodyB, GjkEpa3D.EpaScratch epaScratch)
    {
        var hit = GjkEpa3D.GjkEpa(MakeSupport(bodyA), MakeSupport(bodyB), epaScratch);
        return hit?.Normal ?? Vector3.Zero;
    }

    private static Func<Vector3, Vector3> MakeSupport(RigidBody3 body)
    {
        var t = body.Transform;
        switch (body.Shape)
        {
            case SphereShape3 s:
            {
                var c = t.Position; float r = s.Radius;
                return dir => GjkEpa3D.SupportSphere(c, r, dir);
            }
            case BoxShape3 b:
            {
                var c = t.Position; var he = b.HalfExtents; var rot = t.Rotation;
                return dir => GjkEpa3D.SupportObb(c, he, rot, dir);
            }
            case CapsuleShape3 c:
            {
                var pa = t.TransformPoint(new Vector3(0, -c.HalfLength, 0));
                var pb = t.TransformPoint(new Vector3(0, c.HalfLength, 0));
                float r = c.Radius;
                return dir => GjkEpa3D.SupportCapsule(pa, pb, r, dir);
            }
            case ConvexHullShape3 h:
            {
                var world = h.Vertices.Select(v => t.TransformPoint(v)).ToList();
                return dir => GjkEpa3D.SupportHull(world, dir);
            }
            default:
                return _ => t.Position;
        }
    }

    /// <summary>Clears a body's sleep flag (unknown ids are ignored).</summary>
    /// <summary>
    /// Joint sleep gate: true = skip (both ends frozen: static or
    /// sleeping). A sleeping end with an awake partner wakes first so
    /// joints track moving bodies. Joints3 exposes endpoint ids via
    /// <c>Bodies()</c>, matching Rust joint3.
    /// </summary>
    private bool JointSleepGate(Joint3 joint)
    {
        var (a, b) = joint.Bodies();
        var ba = Bodies[a];
        var bb = Bodies[b];
        bool aFrozen = ba.IsStatic || ba.IsSleeping;
        bool bFrozen = bb.IsStatic || bb.IsSleeping;
        if (aFrozen && bFrozen) return true;
        if (ba.IsSleeping) ba.IsSleeping = false;
        if (bb.IsSleeping) bb.IsSleeping = false;
        return false;
    }

    /// <summary>
    /// Clears a body's sleep flag and cascades through its contact
    /// partners and joint partners, transitively. Unknown ids ignored.
    /// </summary>
    public void WakeBody(int bodyId)
    {
        if (!_idToIndex.TryGetValue(bodyId, out int start)) return;
        Bodies[start].IsSleeping = false;
        bool dirty = true;
        while (dirty)
        {
            dirty = false;
            foreach (var key in ContactCache.Keys)
            {
                int a = key.Item1, b = key.Item2;
                if (a < Bodies.Count && b < Bodies.Count)
                {
                    if (!Bodies[a].IsSleeping && Bodies[b].IsSleeping)
                    { Bodies[b].IsSleeping = false; dirty = true; }
                    else if (!Bodies[b].IsSleeping && Bodies[a].IsSleeping)
                    { Bodies[a].IsSleeping = false; dirty = true; }
                }
            }
            foreach (var joint in Joints)
            {
                var (a, b) = joint.Bodies();
                if (a < Bodies.Count && b < Bodies.Count)
                {
                    if (!Bodies[a].IsSleeping && Bodies[b].IsSleeping)
                    { Bodies[b].IsSleeping = false; dirty = true; }
                    else if (!Bodies[b].IsSleeping && Bodies[a].IsSleeping)
                    { Bodies[a].IsSleeping = false; dirty = true; }
                }
            }
        }
    }
}
