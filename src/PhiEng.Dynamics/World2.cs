using System.Collections.Generic;
using System.Numerics;
using PhiEng.Math;
using PhiEng.Collision;
using PhiEng.Broadphase;
using PhiEng.Quantum;

namespace PhiEng.Dynamics;

/// <summary>
/// Cached contact impulses from the previous frame, used for warm starting.
/// Matched to new contacts by proximity of the contact point; the stored
/// normal guards against reusing impulses after the contact flips.
/// A struct so cache entries live inline in their lists (no per-entry
/// heap object). Mirrors the Rust <c>CachedContact</c>.
/// </summary>
public struct CachedContact
{
    /// <summary>Contact point in world space (last frame).</summary>
    public Vector2 Point { get; set; }
    /// <summary>Contact normal (last frame).</summary>
    public Vector2 Normal { get; set; }
    /// <summary>Accumulated normal impulse j_n.</summary>
    public float NormalImpulse { get; set; }
    /// <summary>Accumulated tangent impulse j_t.</summary>
    public float TangentImpulse { get; set; }
}

/// <summary>
/// Physics world in 2D
/// </summary>
public class World2
{
    /// <summary>Rigid bodies (ids double as indices).</summary>
    public List<RigidBody2> Bodies { get; set; } = new();
    /// <summary>Broad phase spatial hash.</summary>
    public SpatialHash2 Broadphase { get; set; }
    /// <summary>Constraint solver.</summary>
    public Solver Solver { get; set; }
    /// <summary>Gravity.</summary>
    public Vector2 Gravity { get; set; }
    /// <summary>
    /// Gravitational force model + integration scheme (see
    /// <see cref="GravityConfig2"/>). Default is the legacy uniform field
    /// with semi-implicit Euler — bit-identical to pre-gravity behavior.
    /// </summary>
    public GravityConfig2 GravityConfig { get; set; }
    /// <summary>
    /// Ambient force fields — spring (SHM), aerodynamic drag, turbulence
    /// (see <see cref="AmbientConfig2"/>). Default is a no-op.
    /// </summary>
    public AmbientConfig2 Ambient { get; set; } = AmbientConfig2.Default;
    /// <summary>
    /// Continuous collision detection via adaptive substepping (see
    /// <see cref="CcdConfig"/>). Default is a no-op (single legacy substep).
    /// </summary>
    public CcdConfig Ccd { get; set; } = CcdConfig.Default;
    /// <summary>Monotonic simulation clock (s), advanced by Dt each step.</summary>
    public float Time { get; set; }
    /// <summary>Next body id.</summary>
    public int NextBodyId { get; set; }
    /// <summary>Previous frame's contact impulses, keyed by (bodyA, bodyB).</summary>
    public Dictionary<(int, int), List<CachedContact>> ContactCache { get; set; } = new();
    /// <summary>Fixed timestep.</summary>
    public float Dt { get; set; }
    /// <summary>Velocity threshold for sleeping.</summary>
    public float LinearSleepThreshold { get; set; }
    /// <summary>Angular velocity threshold for sleeping.</summary>
    public float AngularSleepThreshold { get; set; }

    /// <summary>Maximum squared distance for matching a new contact to a cached one.</summary>
    public const float ContactMatchDistSq = 0.04f; // (0.2 units)^2

    /// <summary>Runtime validation configuration.</summary>
    public ValidationConfig Validation { get; set; } = ValidationConfig.Default;

    /// <summary>Last validation report.</summary>
    public ValidationReport ValidationReport { get; private set; }

    /// <summary>Joint constraints (see <see cref="Joint2"/>).</summary>
    public List<Joint2> Joints { get; set; } = new();
    /// <summary>Next joint id.</summary>
    public int NextJointId { get; set; }

    /// <summary>SPH fluid state (data only; stepping lives in Rust).</summary>
    public FluidState2 Fluid { get; } = new();

    /// <summary>
    /// Drives a body by the world's quantum field: the body's x position
    /// becomes a Born-rule draw from the wavefunction, and its x
    /// velocity the Bohmian velocity there.
    /// </summary>
    /// <remarks>
    /// A kinematic drive, not a force: the body is placed, not pushed.
    /// The y coordinate is untouched - the field is one-dimensional
    /// along the world's x-axis. Like any placement the drive wakes the
    /// body.
    /// </remarks>
    /// <returns>False when the field is off or the body is unknown.</returns>
    public bool DriveBodyByQuantum(int bodyId)
    {
        if (!Quantum.IsEnabled)
        {
            return false;
        }
        if (bodyId < 0 || bodyId >= Bodies.Count || Bodies[bodyId].Id != bodyId)
        {
            return false;
        }
        float? x = Quantum.SamplePosition();
        if (x == null)
        {
            return false;
        }
        float v = Quantum.VelocityAt(x.Value);
        var body = Bodies[bodyId];
        var transform = body.Transform;
        transform.Position = new System.Numerics.Vector2(x.Value, transform.Position.Y);
        body.Transform = transform;
        var velocity = body.Velocity;
        velocity.X = v;
        body.Velocity = velocity;
        body.IsSleeping = false;
        return true;
    }

    /// <summary>
    /// The quantum wavefunction. Disabled by default, so a world that
    /// never asks for a wavefunction behaves exactly as it did before the
    /// field existed.
    /// </summary>
    public QuantumField Quantum { get; } = new(QuantumConfig.Default);

    // Mapping from body ID to index in Bodies list
    private readonly Dictionary<int, int> _idToIndex = new();

    // Scratch buffers reused every step so the per-frame narrowphase
    // performs no allocations in the steady state.
    private readonly List<ContactConstraint> _constraints = new();

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
    private readonly List<Vector2> _vertsA = new();
    private readonly List<Vector2> _vertsB = new();
    private readonly List<Vector2> _points = new();
    private readonly List<Vector2> _clipTmp = new();
    private readonly HashSet<(int, int)> _liveKeys = new();
    private readonly List<(int, int)> _staleKeys = new();
    /// <summary>The contact events of the last step.</summary>
    private readonly List<ContactEvent2> _contactEvents = new();
    /// <summary>The sensor pairs touching this step, with their contacts.</summary>
    private List<(int, int, Vector2, Vector2)> _sensorTouching = new();
    /// <summary>The sensor pairs that were touching last step, with their contacts.</summary>
    private List<(int, int, Vector2, Vector2)> _lastSensorTouching = new();
    /// <summary>The pairs that already reported a *began* event this step.</summary>
    private readonly HashSet<(uint, uint)> _announcedPairs = new();
    /// <summary>The sensor pairs that ended this step, with their last contact.</summary>
    private readonly List<(int, int, Vector2, Vector2)> _sensorEnded = new();
    /// <summary>The last known contact of every pair, so an *ended* event can say where the pair was.</summary>
    private readonly Dictionary<(int, int), (Vector2, Vector2)> _lastContacts = new();
    /// <summary>The contacts a sensor pair reported.</summary>
    private readonly List<ContactConstraint> _sensorContacts = new();
    private readonly List<bool> _contactTouch = new();

    /// <summary>Empty world with Earth gravity and default solver.</summary>
    public World2()
    {
        Broadphase = new SpatialHash2(1.0f);
        Solver = new Solver();
        Gravity = new Vector2(0.0f, -9.81f);
        GravityConfig = GravityConfig2.Default;
        NextBodyId = 0;
        Dt = 1.0f / 60.0f;
        LinearSleepThreshold = 0.01f;
        AngularSleepThreshold = 0.01f;
        Validation = ValidationConfig.Default;
        ValidationReport = new ValidationReport();
    }

    /// <summary>Adds a rigid body (validates shape and density). Returns the body id.</summary>
    public int AddBody(RigidBody2 body)
    {
        if (body == null) throw new ArgumentNullException(nameof(body));
        if (body.Material == null) throw new ArgumentNullException(nameof(body), "material must be set");
        ShapeValidation.ValidatePosition(body.Transform.Position);
        if (body.Shape != null)
        {
            ShapeValidation.ValidateShape2(body.Shape);
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
    public int AddStaticBody(Vector2 position, Shape2 shape)
    {
        var body = RigidBody2.StaticBody(NextBodyId);
        body.Transform = new Transform2(position, 0f);
        body.SetShape(shape, 1.0f);
        return AddBody(body);
    }

    /// <summary>Adds a dynamic body with mass derived from density.</summary>
    public int AddDynamicBody(Vector2 position, Shape2 shape, float density)
    {
        var body = new RigidBody2(NextBodyId);
        body.Transform = new Transform2(position, 0f);
        body.SetShape(shape, density);
        return AddBody(body);
    }

    /// <summary>
    /// Add a joint. Endpoints are validated against the current body list;
    /// ids are assigned here (id == index invariant, same as bodies).
    /// </summary>
    public int AddJoint(Joint2 joint)
    {
        if (joint == null) throw new ArgumentNullException(nameof(joint));
        if (joint.BodyA < 0 || joint.BodyB < 0
            || joint.BodyA >= Bodies.Count || joint.BodyB >= Bodies.Count)
            throw new ArgumentException($"Unknown joint body id ({joint.BodyA}, {joint.BodyB}).");
        if (joint.BodyA == joint.BodyB)
            throw new ArgumentException("Joint requires two distinct bodies.");
        int id = NextJointId;
        joint.Id = id;
        NextJointId++;
        Joints.Add(joint);
        return id;
    }

    /// <summary>Remove a joint by id (ids double as indices).</summary>
    public void RemoveJoint(int jointId)
    {
        if (jointId < 0 || jointId >= Joints.Count) return;
        Joints.RemoveAt(jointId);
        for (int i = 0; i < Joints.Count; i++) Joints[i].Id = i;
        NextJointId = Joints.Count;
    }

    /// <summary>
    /// Remove a body. Body ids double as list indices (see <see cref="AddBody"/>),
    /// so removal shifts the tail and reassigns ids to preserve the invariant.
    /// Contact-cache keys are remapped through the same shift (entries touching
    /// the removed body are dropped — those contacts are gone — while survivors
    /// keep their warm-start impulses). Pair keys are canonical (min, max), so
    /// the remap re-canonicalizes after substitution. Mirrors Rust <c>remove_body</c>.
    /// </summary>
    public void RemoveBody(int bodyId)
    {
        if (bodyId < 0 || bodyId >= Bodies.Count) return;
        int last = Bodies.Count - 1;
        // Drop joints touching the removed body first (they hold stale ids).
        Joints.RemoveAll(j => j.BodyA == bodyId || j.BodyB == bodyId);
        Bodies.RemoveAt(bodyId);
        _idToIndex.Clear();
        for (int i = 0; i < Bodies.Count; i++)
        {
            Bodies[i].Id = i;
            _idToIndex[i] = i;
        }
        NextBodyId = Bodies.Count;
        // Shift semantics (RemoveAt): survivor joint endpoints above the
        // gap drop by 1 (joints touching the removed body already dropped).
        foreach (var joint in Joints)
        {
            if (!joint.Remap(bodyId)) throw new InvalidOperationException("Joint remap failed after endpoint drop.");
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
                    : (Vector2.Zero, Vector2.Zero);
                _contactEvents.Add(new ContactEvent2(
                    pair.Item1, pair.Item2,
                    ContactEventKind.Ended, lastContact.Item1, lastContact.Item2));
            }
        }

        // The body list shifts down above the gap, so every survivor
        // above it drops by one. The pair keys are remapped through the
        // same shift, and the pairs that touched the removed body are
        // dropped: those contacts are gone.
        var old = new List<((int, int), List<CachedContact>)>();
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
                list = new List<CachedContact>();
                ContactCache[key] = list;
            }
            list.AddRange(entries);
        }
    }

    /// <summary>
    /// Parallel island solve path (see <see cref="IslandManager"/>).
    /// Contact warm-start already ran in <see cref="Step"/> (shared code);
    /// joint warm-start repeats here in joints-vec order, matching the
    /// serial path. Cache write-back stays in <see cref="Step"/>, serial
    /// and canonical, after this returns.
    /// </summary>
    private void SolveParallelPath(List<ContactConstraint> constraints, float dt)
    {
        if (constraints.Count == 0 && Joints.Count == 0)
            return;

        var contactPairs = new List<(int, int)>(constraints.Count);
        foreach (var c in constraints) contactPairs.Add((c.BodyA, c.BodyB));
        var jointPairs = new List<(int, int)>(Joints.Count);
        foreach (var j in Joints) jointPairs.Add((j.BodyA, j.BodyB));
        var islands = IslandManager.BuildIslands(Bodies.Count, contactPairs, jointPairs);

        if (islands.Count <= 1)
        {
            // Degenerate case: exact serial behavior, no thread overhead.
            Solver.Solve(Bodies, constraints, dt);
            if (Joints.Count > 0)
            {
                foreach (var joint in Joints)
                    if (!JointSleepGate(joint)) joint.WarmStart(Bodies);
                for (int it = 0; it < Solver.Iterations; it++)
                    foreach (var joint in Joints)
                        if (!JointSleepGate(joint)) joint.Solve(Bodies, dt);
            }
            return;
        }

        var map = IslandManager.IslandIndexMap(islands, Bodies.Count);
        var islandConstraints = IslandManager.AssignConstraints(islands, contactPairs, map);

        var islandJoints = new List<List<int>>(islands.Count);
        for (int i = 0; i < islands.Count; i++) islandJoints.Add(new List<int>());
        for (int ji = 0; ji < Joints.Count; ji++)
        {
            var j = Joints[ji];
            if (j.BodyA >= 0 && j.BodyB >= 0 && j.BodyA < map.Length && j.BodyB < map.Length
                && map[j.BodyA] >= 0 && map[j.BodyA] == map[j.BodyB])
                islandJoints[map[j.BodyA]].Add(ji);
        }

        // Joint warm-start stays serial (joints vec order, as serial path).
        foreach (var joint in Joints)
            if (!JointSleepGate(joint)) joint.WarmStart(Bodies);

        // Phase 1: contacts per island, in parallel (barrier-shared exit).
        Solver.SolveParallel(Bodies, constraints, islandConstraints, dt);

        // Phase 2: joints per island, in parallel (fixed budget, no exit).
        int iters = Solver.Iterations;
        System.Threading.Tasks.Parallel.For(0, islands.Count, ii =>
        {
            var list = islandJoints[ii];
            for (int it = 0; it < iters; it++)
                foreach (int ji in list)
                    if (!JointSleepGate(Joints[ji])) Joints[ji].Solve(Bodies, dt);
        });
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
    public List<uint> OverlapAabb(AABB2 aabb, QueryFilter filter)
    {
        var result = new List<uint>();
        foreach (var body in Bodies)
        {
            if (!filter.Accepts(body)) continue;
            var (min, max) = GetBodyAABB(body);
            if (min.X <= aabb.Max.X && max.X >= aabb.Min.X
                && min.Y <= aabb.Max.Y && max.Y >= aabb.Min.Y)
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
    /// Every body whose shape contains <paramref name="point"/>, in
    /// ascending id order.
    /// </summary>
    /// <remarks>
    /// Sensors are included: a point test is a pick or trigger query, and
    /// a trigger volume is exactly what it should find. (The ray cast
    /// skips sensors because a ray passes through them.)
    /// </remarks>
    public int[] PointTest(Vector2 point, QueryFilter filter)
    {
        var hits = new List<int>();
        foreach (var body in Bodies)
        {
            if (!filter.Accepts(body)) continue;
            if (body.Shape == null) continue;
            if (ShapeQuery2.PointInShape(
                    body.Shape, body.Transform.Position, body.Transform.Rotation, point))
            {
                hits.Add(body.Id);
            }
        }
        return hits.ToArray();
    }

    /// <summary>The nearest body surface to <paramref name="point"/>, within <paramref name="maxDistance"/>.</summary>
    /// <remarks>
    /// A point inside a shape wins (distance zero) over any outside
    /// surface; ties keep the lower body id. Sensors are included, as in
    /// <see cref="PointTest"/>.
    /// </remarks>
    public WorldClosestHit2? ClosestPoint(Vector2 point, QueryFilter filter, float maxDistance)
    {
        float max = float.IsFinite(maxDistance) ? MathF.Max(maxDistance, 0.0f) : float.MaxValue;
        WorldClosestHit2? best = null;
        foreach (var body in Bodies)
        {
            if (!filter.Accepts(body)) continue;
            if (body.Shape == null) continue;
            var hit = ShapeQuery2.ClosestPoint(
                body.Shape, body.Transform.Position, body.Transform.Rotation, point);
            if (hit == null) continue;
            var h = hit.Value;
            if (!h.Inside && h.Distance > max) continue;
            if (best == null || h.Distance < best.Value.Distance)
            {
                best = new WorldClosestHit2
                {
                    Body = body.Id,
                    Inside = h.Inside,
                    Point = h.Point,
                    Normal = h.Normal,
                    Distance = h.Distance,
                };
            }
        }
        return best;
    }

    /// <summary>Every body that overlaps <paramref name="shape"/> at the given pose, in ascending id order.</summary>
    /// <remarks>
    /// Sensors are included, as in <see cref="PointTest"/>: a shape query
    /// is a pick/trigger query. Touching counts as overlapping. O(body
    /// count): exact against every body, with no broad-phase prefilter yet.
    /// </remarks>
    public int[] OverlapShape(Shape2 shape, Vector2 position, Matrix3x2 rotation, QueryFilter filter)
    {
        var hits = new List<int>();
        foreach (var body in Bodies)
        {
            if (!filter.Accepts(body)) continue;
            if (body.Shape == null) continue;
            if (ShapeQuery2.Overlap(
                    shape, position, rotation,
                    body.Shape, body.Transform.Position, body.Transform.Rotation))
            {
                hits.Add(body.Id);
            }
        }
        return hits.ToArray();
    }

    /// <summary>
    /// Cast a ray into the world and return the first thing it hits.
    /// </summary>
    /// <remarks>
    /// The ray is tested against every body's current shape; the body
    /// that is hit first wins. A body the filter does not accept is not
    /// considered. A ray that starts inside a shape does not hit it.
    /// </remarks>
    public (uint Body, float T, Vector2 Point, Vector2 Normal)? Raycast(
        Vector2 origin, Vector2 direction, float maxT, QueryFilter filter)
    {
        (uint Body, float T, Vector2 Point, Vector2 Normal)? best = null;
        foreach (var body in Bodies)
        {
            if (!filter.Accepts(body)) continue;
            // A sensor is a trigger volume, not a solid: a ray passes
            // through it.
            if (body.IsSensor) continue;
            if (body.Shape == null) continue;
            Vector2 point;
            Vector2 normal;
            float t;
            bool hit;
            switch (body.Shape)
            {
                case CircleShape circle:
                    hit = Collision.Query.RayCircle(
                        origin, direction, body.Transform.Position, circle.Radius,
                        out t, out point, out normal);
                    break;
                case BoxShape2 box:
                    hit = Collision.Query.RayObb(
                        origin, direction, body.Transform.Position, body.Transform.Rotation,
                        box.HalfExtents, out t, out point, out normal);
                    break;
                case PolygonShape polygon:
                    var world = new Vector2[polygon.Vertices.Count];
                    for (int i = 0; i < polygon.Vertices.Count; i++)
                    {
                        world[i] = body.Transform.TransformPoint(polygon.Vertices[i]);
                    }
                    hit = Collision.Query.RayPolygon(
                        origin, direction, world, out t, out point, out normal);
                    break;
                default:
                    hit = false;
                    t = 0.0f;
                    point = Vector2.Zero;
                    normal = Vector2.Zero;
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
    public IReadOnlyList<ContactEvent2> ContactEvents => _contactEvents;

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

        // 7e. The quantum wavefunction advances with the world's step. The
        // field is off by default, so this is a branch and not a
        // behaviour change: a world that never asks for a wavefunction
        // keeps its trajectory to the last bit.
        if (Quantum.IsEnabled)
        {
            Quantum.Step(dt);
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
            float eff = body.Velocity.Length() + System.Math.Abs(body.AngularVelocity) * extent;
            peak = System.Math.Max(peak, eff);
        }
        return peak;
    }

    private void StepInner(float dt)
    {
        // 1. Gravity kick (see GravityIntegration): uniform background
        // field plus Central/N-body terms per GravityConfig, one
        // full-step kick (Euler) or the first half-kick (Verlet, with
        // linear damping riding as symmetric drag).
        bool verlet = GravityConfig.Integrator == Integrator.VelocityVerlet;
        float h = verlet ? dt * 0.5f : dt;
        GravityIntegration.Kick2(Bodies, GravityConfig, Gravity, h, verlet);

        // 1a. Ambient forces (spring/SHM, drag, turbulence), same slot and
        // Verlet split as the gravity kick above (see AmbientIntegration).
        if (Ambient.IsActive())
        {
            AmbientIntegration.Kick2(Bodies, Ambient, Time, h);
        }

        // 1b. Fluid phase - SPH particles and two-way body coupling. Runs
        // before broadphase so the rigid pipeline observes post-fluid
        // velocities; a no-op when the fluid holds no particles.
        FluidIntegration.StepFluid2(Fluid, Bodies, Gravity, Ambient, Time, dt);

        // The events of the last step are the caller's to read; they are
        // cleared when the next step starts.
        _contactEvents.Clear();

        // 2. Broad phase - find potential collision pairs (into scratch).
        UpdateBroadphase();
        _pairs.Clear();
        Broadphase.CollectPairs(_pairs);

        // 3. Narrow phase - test actual collisions (into scratch).
        _constraints.Clear();
        var constraints = _constraints;
        _vertsA.Clear();
        _vertsB.Clear();
        _sensorContacts.Clear();
        foreach (var (a, b) in _pairs)
        {
            // A sensor reports contacts but never pushes: its contacts
            // go to the sensor list, not to the solver.
            if (Bodies[a].IsSensor || Bodies[b].IsSensor)
            {
                TestCollisionsInto(a, b, _sensorContacts, _vertsA, _vertsB, _points, _clipTmp);
            }
            else
            {
                TestCollisionsInto(a, b, constraints, _vertsA, _vertsB, _points, _clipTmp);
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

        // 3b. Warm-start: seed each constraint with cached impulses from the
        // previous frame (matched by contact-point proximity + normal agreement).
        // A pair with no cache entry is a NEW contact: any sleeping party
        // wakes (with cascade) so it can respond to the newcomer. Every
        // constrained body is marked touched (sleep-support tracking).
        if (_contactTouch.Count != Bodies.Count)
        {
            _contactTouch.Clear();
            for (int ti = 0; ti < Bodies.Count; ti++) _contactTouch.Add(false);
        }
        for (int ti = 0; ti < _contactTouch.Count; ti++) _contactTouch[ti] = false;
        for (int ci = 0; ci < constraints.Count; ci++)
        {
            var constraint = constraints[ci];
            _contactTouch[constraint.BodyA] = true;
            _contactTouch[constraint.BodyB] = true;
            bool isNew = !ContactCache.ContainsKey((constraint.BodyA, constraint.BodyB));
            // A pair that was not touching last step began touching this
            // step: that is the event a trigger wants.
            if (isNew && CollectContactEvents)
            {
                var pair = EventPair(constraint.BodyA, constraint.BodyB);
                // One event per *pair*: a pair that touches at two points
                // began touching once.
                if (_announcedPairs.Add((pair.Item1, pair.Item2)))
                {
                    _contactEvents.Add(new ContactEvent2(
                        pair.Item1, pair.Item2,
                        ContactEventKind.Began, constraint.Point, constraint.Normal));
                }
            }
            if (isNew
                && (Bodies[constraint.BodyA].IsSleeping || Bodies[constraint.BodyB].IsSleeping))
            {
                WakeBody(constraint.BodyA);
                WakeBody(constraint.BodyB);
            }
            if (ContactCache.TryGetValue((constraint.BodyA, constraint.BodyB), out var entries))
            {
                CachedContact? best = null;
                float bestDist = ContactMatchDistSq;
                foreach (var entry in entries)
                {
                    // Skip entries whose contact flipped (impulses would push the wrong way).
                    if (Vector2.Dot(entry.Normal, constraint.Normal) <= 0.0f)
                    {
                        continue;
                    }
                    float d = (entry.Point - constraint.Point).LengthSquared();
                    if (d < bestDist)
                    {
                        bestDist = d;
                        best = entry;
                    }
                }
                if (best != null)
                {
                    var match = best.Value;
                    constraint.NormalImpulse = match.NormalImpulse;
                    constraint.TangentImpulse = match.TangentImpulse;
                    constraints[ci] = constraint;
                }
            }
        }

        // Validate constraints before solving
        var constraintReport = Validation2D.ValidateConstraints(Bodies, constraints, Validation);
        ValidationReport.Merge(constraintReport);

        // 4. Solve constraints (+ joints): serial reference path vs
        // parallel island path (bit-identical; see IslandManager).
        if (Solver.Parallel)
            SolveParallelPath(constraints, dt);
        else
        {
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
        }

        // 4c. Write back accumulated impulses for next frame's warm start.
        // Rebuilt every frame so stale pairs/contacts are evicted
        // automatically. Entry lists are recycled from the previous
        // cache when the pair survives, so steady-state contact keeps
        // its allocation instead of reallocating every frame.
        // Reuse the live dictionary object itself: clear every entry
        // list in place, refill surviving pairs into their own lists,
        // then evict stale (still-empty) keys. No allocations in the
        // steady state.
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
        foreach (var constraint in constraints)
        {
            var key = (constraint.BodyA, constraint.BodyB);
            _liveKeys.Add(key);
            if (!ContactCache.TryGetValue(key, out var list) || list == null)
            {
                list = new List<CachedContact>(2);
                ContactCache[key] = list;
            }
            list.Add(new CachedContact
            {
                Point = constraint.Point,
                Normal = constraint.Normal,
                NormalImpulse = constraint.NormalImpulse,
                TangentImpulse = constraint.TangentImpulse
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
                        : (Vector2.Zero, Vector2.Zero);
                    _contactEvents.Add(new ContactEvent2(
                        pair.Item1, pair.Item2,
                        ContactEventKind.Ended, lastContact.Item1, lastContact.Item2));
                }
                // A vanished pair is lost contact: wake any sleeping party
                // (loss of support must re-evaluate sleep).
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
            // 5v. Verlet drift (positions only — no linear damping) plus
            // the second half-kick with forces recomputed at the new
            // positions. The contact solver above ran unchanged.
            // Sleeping bodies are skipped: frozen until woken.
            foreach (var body in Bodies)
            {
                if (!body.IsSleeping) body.IntegrateDrift(dt);
            }
            GravityIntegration.Kick2(Bodies, GravityConfig, Gravity, dt * 0.5f, true);
            if (Ambient.IsActive())
            {
                AmbientIntegration.Kick2(Bodies, Ambient, Time, dt * 0.5f);
            }
        }
        else
        {
            // 5. Integrate motion (sleeping bodies are frozen until woken).
            foreach (var body in Bodies)
            {
                if (!body.IsSleeping) body.Integrate(dt);
            }
        }

        // 5c. Nuclear decay and fission: fills a per-body heat budget
        // (and an ionizing-radiation budget) from the nuclides the
        // bodies carry. Runs before the thermal phase, so the thermal
        // radiation sees the heat in the same step. It modifies
        // temperature only (no velocity change), so it neither wakes nor
        // is gated by sleep.
        if (_nuclear.Config.IsActive())
        {
            NuclearIntegration.StepNuclear2(Bodies, _nuclear.Config, _nuclear.Scratch, dt);
            // 5c-2. The nuclear heat raises the body temperatures. The
            // clamps come from the thermal configuration, so the bounds
            // are the same ones the thermal phase enforces; the rate
            // clamp is deliberately not applied, because a nuclear
            // source has a physical rate.
            NuclearIntegration.ApplyNuclearHeat2(Bodies, _nuclear.Scratch, _thermal.Config);
        }

        // 5d. Thermal transfer (conduction / radiation / convection),
        // then 5e. Thermal expansion. The phase runs after integration
        // and before the sleep check, matching the Rust core. It does not
        // modify velocity, so it neither wakes sleeping bodies nor is
        // affected by sleep.
        ThermalIntegration.StepThermal2(Bodies, constraints, _thermal.Config, _thermal, dt);
        ThermalIntegration.ApplyExpansion2(Bodies, _thermal.Config);

        // 6+7. Validate bodies + update sleep in a single fused sweep
        // (bit-identical to validate-all-then-sleep-all; mirrors Rust).
        // Sleep needs slowness AND support (a current contact): ballistic
        // apexes and drifting bodies never sleep, so no wake/sleep
        // history is needed and save/load stays exact. Sleeping bodies
        // have zeroed velocities every step (the solver may write
        // reaction impulses to a frozen body mid-step).
        float linSleep = LinearSleepThreshold;
        float angSleep = AngularSleepThreshold;
        for (int i = 0; i < Bodies.Count; i++)
        {
            var bodyReport = Validation2D.ValidateBody(Bodies[i], Validation);
            ValidationReport.Merge(bodyReport);
            var body = Bodies[i];
            if (body.IsStatic) continue;
            if (body.IsSleeping)
            {
                body.Velocity = Vector2.Zero;
                body.AngularVelocity = 0.0f;
                continue;
            }
            if (body.ShouldSleep(linSleep, angSleep) && _contactTouch[i])
            {
                body.IsSleeping = true;
                body.Velocity = Vector2.Zero;
                body.AngularVelocity = 0.0f;
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

    private (Vector2, Vector2) GetBodyAABB(RigidBody2 body)
    {
        if (body.Shape == null)
        {
            return (body.Transform.Position, body.Transform.Position);
        }

        switch (body.Shape.Type)
        {
            case Shape2Type.Circle:
                var circle = body.Shape as CircleShape;
                if (circle != null)
                {
                    float r = circle.Radius;
                    return (body.Transform.Position - new Vector2(r, r),
                            body.Transform.Position + new Vector2(r, r));
                }
                break;
            case Shape2Type.Box:
                var box = body.Shape as BoxShape2;
                if (box != null)
                {
                    // Exact AABB from the rotated corners (an axis-aligned
                    // pos±he box would MISS pairs for rotated boxes).
                    // Unrolled corners: no array allocation.
                    var he = box.HalfExtents;
                    var t = body.Transform;
                    Vector2 min = t.TransformPoint(new Vector2(-he.X, -he.Y));
                    Vector2 max = min;
                    Vector2 p1 = t.TransformPoint(new Vector2(he.X, -he.Y));
                    Vector2 p2 = t.TransformPoint(new Vector2(he.X, he.Y));
                    Vector2 p3 = t.TransformPoint(new Vector2(-he.X, he.Y));
                    min = Vector2.Min(min, p1);
                    max = Vector2.Max(max, p1);
                    min = Vector2.Min(min, p2);
                    max = Vector2.Max(max, p2);
                    min = Vector2.Min(min, p3);
                    max = Vector2.Max(max, p3);
                    return (min, max);
                }
                break;
            case Shape2Type.Polygon:
                var polyShape = body.Shape as PolygonShape;
                if (polyShape != null && polyShape.Vertices.Count > 0)
                {
                    Vector2 min = body.Transform.TransformPoint(polyShape.Vertices[0]);
                    Vector2 max = min;
                    for (int i = 1; i < polyShape.Vertices.Count; i++)
                    {
                        Vector2 t = body.Transform.TransformPoint(polyShape.Vertices[i]);
                        min = Vector2.Min(min, t);
                        max = Vector2.Max(max, t);
                    }
                    return (min, max);
                }
                break;
        }

        return (body.Transform.Position, body.Transform.Position);
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
                    _contactEvents.Add(new ContactEvent2(
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
            _contactEvents.Add(new ContactEvent2(
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
    private Vector2 SensorPoint((int, int) key)
    {
        foreach (var c in _sensorContacts)
            if (c.BodyA == key.Item1 && c.BodyB == key.Item2) return c.Point;
        return Vector2.Zero;
    }

    /// <summary>The contact normal of a sensor pair, if it has one.</summary>
    private Vector2 SensorNormal((int, int) key)
    {
        foreach (var c in _sensorContacts)
            if (c.BodyA == key.Item1 && c.BodyB == key.Item2) return c.Normal;
        return Vector2.Zero;
    }

    private void TestCollisionsInto(
        int a, int b,
        List<ContactConstraint> output,
        List<Vector2> vertsA, List<Vector2> vertsB,
        List<Vector2> points, List<Vector2> clipTmp)
    {
        // Convert body IDs to indices
        if (!_idToIndex.TryGetValue(a, out int idxA) || !_idToIndex.TryGetValue(b, out int idxB))
        {
            return;
        }
        // Static-static pairs can never produce an impulse; skip the
        // narrowphase. Bit-identical (their constraints would solve to zero).
        if (Bodies[idxA].IsStatic && Bodies[idxB].IsStatic) return;
        // Collision layers: each body must accept the other's
        // layer. The default filter accepts everything, so an
        // unconfigured scene is unaffected.
        if (!CollisionFilter.Allows(Bodies[idxA], Bodies[idxB])) return;

        var bodyA = Bodies[idxA];
        var bodyB = Bodies[idxB];

        vertsA.Clear();
        vertsB.Clear();
        var typeA = WriteShapeVertices(bodyA, vertsA);
        var typeB = WriteShapeVertices(bodyB, vertsB);

        // Dispatch based on shape types
        switch (typeA, typeB)
        {
            case (Shape2Type.Circle, Shape2Type.Circle):
                CollideCircleCircleInto(bodyA, bodyB, a, b, output);
                break;
            case (Shape2Type.Circle, Shape2Type.Box):
                CollideCircleBoxInto(bodyA, a, bodyB, b, vertsB, output);
                break;
            case (Shape2Type.Box, Shape2Type.Circle):
                CollideCircleBoxInto(bodyB, b, bodyA, a, vertsA, output);
                break;
            case (Shape2Type.Box, Shape2Type.Box):
                CollideBoxBoxInto(bodyA, bodyB, a, b, vertsA, vertsB, output, points, clipTmp);
                break;
            case (Shape2Type.Circle, Shape2Type.Polygon):
                CollideCirclePolygonInto(bodyA, a, bodyB, b, vertsB, output);
                break;
            case (Shape2Type.Polygon, Shape2Type.Circle):
                CollideCirclePolygonInto(bodyB, b, bodyA, a, vertsA, output);
                break;
            case (Shape2Type.Box, Shape2Type.Polygon):
                CollideBoxPolygonInto(bodyA, a, bodyB, b, vertsA, vertsB, output, points, clipTmp);
                break;
            case (Shape2Type.Polygon, Shape2Type.Box):
                CollideBoxPolygonInto(bodyB, b, bodyA, a, vertsB, vertsA, output, points, clipTmp);
                break;
            case (Shape2Type.Polygon, Shape2Type.Polygon):
                CollidePolygonPolygonInto(bodyA, bodyB, a, b, vertsA, vertsB, output, points, clipTmp);
                break;
        }
    }

    /// <summary>
    /// Push one constraint per contact point in <paramref name="points"/>
    /// (all sharing the manifold normal/penetration, up to 2), or a
    /// single midpoint constraint when empty.
    /// </summary>
    private static void PushConstraintsForPoints(
        int bodyA, int bodyB,
        Vector2 fallbackCenter,
        List<Vector2> points,
        Vector2 normal,
        float penetration,
        float friction,
        float restitution,
        List<ContactConstraint> output)
    {
        if (points.Count == 0)
        {
            output.Add(new ContactConstraint
            {
                BodyA = bodyA,
                BodyB = bodyB,
                Point = fallbackCenter + normal * (penetration * 0.5f),
                Normal = normal,
                Penetration = penetration,
                Friction = friction,
                Restitution = restitution,
                NormalImpulse = 0.0f,
                TangentImpulse = 0.0f
            });
            return;
        }
        int count = System.Math.Min(2, points.Count);
        for (int i = 0; i < count; i++)
        {
            output.Add(new ContactConstraint
            {
                BodyA = bodyA,
                BodyB = bodyB,
                Point = points[i],
                Normal = normal,
                Penetration = penetration,
                Friction = friction,
                Restitution = restitution,
                NormalImpulse = 0.0f,
                TangentImpulse = 0.0f
            });
        }
    }

    private static void PushConstraint(
        int bodyA, int bodyB,
        Vector2 point, Vector2 normal, float penetration,
        float friction, float restitution,
        List<ContactConstraint> output)
    {
        output.Add(new ContactConstraint
        {
            BodyA = bodyA,
            BodyB = bodyB,
            Point = point,
            Normal = normal,
            Penetration = penetration,
            Friction = friction,
            Restitution = restitution,
            NormalImpulse = 0.0f,
            TangentImpulse = 0.0f
        });
    }

    private static Shape2Type WriteShapeVertices(RigidBody2 body, List<Vector2> output)
    {
        output.Clear();
        if (body.Shape == null)
        {
            return Shape2Type.Circle;
        }

        switch (body.Shape.Type)
        {
            case Shape2Type.Circle:
                // Circles need no vertices (circle-circle and
                // circle-polygon use analytic handling).
                return Shape2Type.Circle;

            case Shape2Type.Box:
                var box = body.Shape as BoxShape2;
                if (box != null)
                {
                    var he = box.HalfExtents;
                    if (output.Capacity < 4) output.Capacity = 4;
                    output.Add(body.Transform.TransformPoint(new Vector2(-he.X, -he.Y)));
                    output.Add(body.Transform.TransformPoint(new Vector2(he.X, -he.Y)));
                    output.Add(body.Transform.TransformPoint(new Vector2(he.X, he.Y)));
                    output.Add(body.Transform.TransformPoint(new Vector2(-he.X, he.Y)));
                    return Shape2Type.Box;
                }
                break;

            case Shape2Type.Polygon:
                var poly = body.Shape as PolygonShape;
                if (poly != null)
                {
                    if (output.Capacity < poly.Vertices.Count)
                        output.Capacity = poly.Vertices.Count;
                    foreach (var v in poly.Vertices)
                        output.Add(body.Transform.TransformPoint(v));
                    return Shape2Type.Polygon;
                }
                break;
        }
        return Shape2Type.Circle;
    }

    private static void PushSingleConstraint(
        int bodyA, int bodyB, Vector2 point, Vector2 normal, float penetration,
        float friction, float restitution, List<ContactConstraint> output)
        => PushConstraint(bodyA, bodyB, point, normal, penetration, friction, restitution, output);

    private static void CollideCircleCircleInto(RigidBody2 bodyA, RigidBody2 bodyB, int a, int b, List<ContactConstraint> output)
    {
        if (bodyA.Shape is CircleShape circleA && bodyB.Shape is CircleShape circleB)
        {
            Vector2 delta = bodyB.Transform.Position - bodyA.Transform.Position;
            float distance = delta.Length();
            float totalRadius = circleA.Radius + circleB.Radius;

            if (distance < totalRadius)
            {
                Vector2 normal = distance > 0.0f ? delta / distance : Vector2.UnitX;
                float penetration = totalRadius - distance;
                Vector2 contactPoint = bodyA.Transform.Position + normal * circleA.Radius;

                PushSingleConstraint(
                    a, b, contactPoint, normal, penetration,
                    PhysicsMaterial.CombineFriction(bodyA.Material.Friction, bodyB.Material.Friction),
                    PhysicsMaterial.CombineRestitution(bodyA.Material.Restitution, bodyB.Material.Restitution),
                    output);
            }
        }
    }

    private static void CollideCircleBoxInto(
        RigidBody2 circleBody, int circleId, RigidBody2 boxBody, int boxId,
        List<Vector2> boxVerts, List<ContactConstraint> output)
    {
        if (circleBody.Shape is CircleShape circle && boxBody.Shape is BoxShape2)
        {
            if (SAT.CirclePolygon(circleBody.Transform.Position, circle.Radius, boxVerts) is { } result)
            {
                var normal = SAT.OrientNormal(result.Normal, circleBody.Transform.Position, SAT.PolygonCentroid(boxVerts));
                var penetration = result.Penetration;
                var contactPoint = circleBody.Transform.Position + normal * circle.Radius;
                PushSingleConstraint(
                    circleId, boxId, contactPoint, normal, penetration,
                    PhysicsMaterial.CombineFriction(circleBody.Material.Friction, boxBody.Material.Friction),
                    PhysicsMaterial.CombineRestitution(circleBody.Material.Restitution, boxBody.Material.Restitution),
                    output);
            }
        }
    }

    private static void CollideBoxBoxInto(
        RigidBody2 bodyA, RigidBody2 bodyB, int a, int b,
        List<Vector2> vertsA, List<Vector2> vertsB,
        List<ContactConstraint> output, List<Vector2> points, List<Vector2> clipTmp)
    {
        if (SAT.PolygonPolygon(vertsA, vertsB) is { } result)
        {
            var normal = SAT.OrientNormal(result.Normal, SAT.PolygonCentroid(vertsA), SAT.PolygonCentroid(vertsB));
            var penetration = result.Penetration;
            points.Clear();
            SAT.PolygonPolygonContactPointsInto(vertsA, vertsB, normal, points, clipTmp);

            PushConstraintsForPoints(
                a, b,
                bodyA.Transform.Position, points, normal, penetration,
                PhysicsMaterial.CombineFriction(bodyA.Material.Friction, bodyB.Material.Friction),
                PhysicsMaterial.CombineRestitution(bodyA.Material.Restitution, bodyB.Material.Restitution),
                output);
        }
    }

    private static void CollideCirclePolygonInto(
        RigidBody2 circleBody, int circleId, RigidBody2 polyBody, int polyId,
        List<Vector2> polyVerts, List<ContactConstraint> output)
    {
        if (circleBody.Shape is CircleShape circle)
        {
            if (SAT.CirclePolygon(circleBody.Transform.Position, circle.Radius, polyVerts) is { } result)
            {
                var normal = SAT.OrientNormal(result.Normal, circleBody.Transform.Position, SAT.PolygonCentroid(polyVerts));
                var penetration = result.Penetration;
                var contactPoint = circleBody.Transform.Position + normal * circle.Radius;
                PushSingleConstraint(
                    circleId, polyId, contactPoint, normal, penetration,
                    PhysicsMaterial.CombineFriction(circleBody.Material.Friction, polyBody.Material.Friction),
                    PhysicsMaterial.CombineRestitution(circleBody.Material.Restitution, polyBody.Material.Restitution),
                    output);
            }
        }
    }

    private static void CollideBoxPolygonInto(
        RigidBody2 boxBody, int boxId, RigidBody2 polyBody, int polyId,
        List<Vector2> boxVerts, List<Vector2> polyVerts,
        List<ContactConstraint> output, List<Vector2> points, List<Vector2> clipTmp)
    {
        if (SAT.PolygonPolygon(boxVerts, polyVerts) is { } result)
        {
            var normal = SAT.OrientNormal(result.Normal, SAT.PolygonCentroid(boxVerts), SAT.PolygonCentroid(polyVerts));
            var penetration = result.Penetration;
            points.Clear();
            SAT.PolygonPolygonContactPointsInto(boxVerts, polyVerts, normal, points, clipTmp);

            PushConstraintsForPoints(
                boxId, polyId,
                boxBody.Transform.Position, points, normal, penetration,
                PhysicsMaterial.CombineFriction(boxBody.Material.Friction, polyBody.Material.Friction),
                PhysicsMaterial.CombineRestitution(boxBody.Material.Restitution, polyBody.Material.Restitution),
                output);
        }
    }

    private static void CollidePolygonPolygonInto(
        RigidBody2 bodyA, RigidBody2 bodyB, int a, int b,
        List<Vector2> vertsA, List<Vector2> vertsB,
        List<ContactConstraint> output, List<Vector2> points, List<Vector2> clipTmp)
    {
        if (SAT.PolygonPolygon(vertsA, vertsB) is { } result)
        {
            var normal = SAT.OrientNormal(result.Normal, SAT.PolygonCentroid(vertsA), SAT.PolygonCentroid(vertsB));
            var penetration = result.Penetration;
            points.Clear();
            SAT.PolygonPolygonContactPointsInto(vertsA, vertsB, normal, points, clipTmp);

            PushConstraintsForPoints(
                a, b,
                bodyA.Transform.Position, points, normal, penetration,
                PhysicsMaterial.CombineFriction(bodyA.Material.Friction, bodyB.Material.Friction),
                PhysicsMaterial.CombineRestitution(bodyA.Material.Restitution, bodyB.Material.Restitution),
                output);
        }
    }

    /// <summary>
    /// Joint sleep gate: true = skip (both ends frozen: static or
    /// sleeping). A sleeping end with an awake partner wakes first so
    /// joints track moving bodies. Mirrors Rust joint_sleep_gate.
    /// </summary>
    private bool JointSleepGate(Joint2 joint)
    {
        var a = Bodies[joint.BodyA];
        var b = Bodies[joint.BodyB];
        bool aFrozen = a.IsStatic || a.IsSleeping;
        bool bFrozen = b.IsStatic || b.IsSleeping;
        if (aFrozen && bFrozen) return true;
        if (a.IsSleeping) a.IsSleeping = false;
        if (b.IsSleeping) b.IsSleeping = false;
        return false;
    }

    /// <summary>
    /// Clears a body's sleep flag and cascades through its contact
    /// partners and joint partners, transitively. The woken set is a
    /// fixpoint over idempotent flags, so iteration order cannot affect
    /// determinism. Unknown ids are ignored.
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
                int a = joint.BodyA, b = joint.BodyB;
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
