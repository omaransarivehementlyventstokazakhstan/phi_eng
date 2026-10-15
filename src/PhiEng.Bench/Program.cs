using System.Diagnostics;
using System.Numerics;
using PhiEng.Dynamics;

namespace PhiEng.Bench;

/// <summary>
/// Managed-engine benchmark: pile / stack / joints scenarios with per-step
/// timings, GC census, pair/contact census, and trajectory hashes.
///
/// The numbers feed the struct-of-arrays decision: if body-pointer chasing
/// dominates step cost at scale, SoA proceeds; otherwise AoS stays.
/// Methodology mirrors Rust bench_large (medians over repeated windows
/// after settling; always read time WITH pair/contact counts).
///
/// Usage:
///   dotnet run --project src/PhiEng.Bench -c Release [-- pile|pile3d|stack|joints|islands|all] [--frames N]
/// </summary>
internal static class Program
{
    private const float Dt = 1.0f / 60.0f;

    private static int Main(string[] args)
    {
        string scenario = args.Length > 0 ? args[0].ToLowerInvariant() : "all";
        int frames = 300;
        for (int i = 0; i + 1 < args.Length; i++)
        {
            if (args[i] == "--frames" && int.TryParse(args[i + 1], out int f) && f > 0)
                frames = f;
        }

        bool ok = true;
        if (scenario is "all" or "pile") ok &= RunPile(frames);
        if (scenario is "all" or "pile3d") ok &= RunPile3D(frames);
        if (scenario is "all" or "stack") ok &= RunStack(frames);
        if (scenario is "all" or "joints") ok &= RunJoints(frames);
        if (scenario is "all" or "islands") ok &= RunIslands(frames);
        return ok ? 0 : 1;
    }

    /// <summary>Builds a world and steps it until settled, then measures.</summary>
    private static (double MsPerStep, ulong Hash, int Pairs, int Contacts, double Gc0)
        Measure(World2 world, int frames, int warmupIters = 8)
    {
        world.Solver.Iterations = warmupIters;
        for (int i = 0; i < 60; i++) world.Step(Dt); // settle briefly

        const int windows = 5;
        var samples = new double[windows];
        for (int s = 0; s < windows; s++)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < frames; i++) world.Step(Dt);
            sw.Stop();
            samples[s] = sw.Elapsed.TotalMilliseconds / frames;
        }
        Array.Sort(samples);

        long gen0 = GC.CollectionCount(0);
        int pairs = 0, contacts = 0;
        for (int i = 0; i < frames; i++)
        {
            world.Step(Dt);
            pairs += world.ContactCache.Count;
            foreach (var list in world.ContactCache.Values) contacts += list.Count;
        }
        double gc0 = (GC.CollectionCount(0) - gen0) / (double)frames;

        return (samples[windows / 2], HashWorld(world),
            pairs / frames, contacts / frames, gc0);
    }

    private static ulong HashWorld(World2 world)
    {
        ulong h = 0xcbf29ce484222325UL;
        foreach (var body in world.Bodies)
        {
            foreach (float v in new[]
            {
                body.Transform.Position.X, body.Transform.Position.Y,
                body.Velocity.X, body.Velocity.Y,
                body.AngularVelocity, body.RotationAngle
            })
            {
                h ^= (ulong)BitConverter.SingleToInt32Bits(v);
                h *= 0x100000001b3UL;
            }
        }
        return h;
    }

    private static bool RunPile(int frames)
    {
        const int bodies = 1500;
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(30.0f, 0.5f)));
        for (int i = 0; i < bodies; i++)
        {
            float x = -14.0f + (i % 30) * 0.95f;
            float y = 2.0f + (i / 30) * 0.75f;
            if (i % 3 == 0)
                world.AddDynamicBody(new Vector2(x, y), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
            else
                world.AddDynamicBody(new Vector2(x, y), new CircleShape(0.3f), 1.0f);
        }
        for (int i = 0; i < 900; i++) world.Step(Dt); // settle into a pile

        foreach (int iters in new[] { 1, 8 })
        {
            world.Solver.Iterations = iters;
            var m = Measure(world, frames, iters);
            Console.WriteLine(
                $"[pile] iters={iters}: {m.MsPerStep:F3} ms/step " +
                $"pairs={m.Pairs} contacts={m.Contacts} gc0={m.Gc0:F3}/step hash=0x{m.Hash:x}");
        }

        // Resting steady state (single box on floor): the allocation floor.
        var calm = new World2();
        calm.Gravity = new Vector2(0.0f, -9.81f);
        calm.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        calm.AddDynamicBody(new Vector2(0.0f, 0.2f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        for (int i = 0; i < 600; i++) calm.Step(Dt);
        var c = Measure(calm, frames);
        Console.WriteLine(
            $"[pile-resting] {c.MsPerStep:F4} ms/step gc0={c.Gc0:F4}/step hash=0x{c.Hash:x}");
        return true;
    }

    /// <summary>
    /// 3D pile control column: 400 bodies over a static floor. Mirrors the
    /// Rust bench_large [pile3d] section (same grid, same settle). 3D was
    /// never benched at scale; this is the baseline for any future
    /// RigidBody3/Solver3D layout work.
    /// </summary>
    private static bool RunPile3D(int frames)
    {
        var world = new World3();
        world.Gravity = new Vector3(0.0f, -9.81f, 0.0f);
        world.AddStaticBody(new Vector3(0.0f, -1.0f, 0.0f),
            new BoxShape3(new Vector3(30.0f, 0.5f, 30.0f)));
        for (int i = 0; i < 400; i++)
        {
            float x = -9.0f + (i % 20) * 0.95f;
            float z = -4.5f + ((i / 20) % 10) * 0.95f;
            float y = 2.0f + (i / 200) * 0.75f;
            Shape3 shape = i % 3 == 0
                ? new BoxShape3(new Vector3(0.3f, 0.3f, 0.3f))
                : new SphereShape3(0.3f);
            world.AddDynamicBody(new Vector3(x, y, z), shape, 1.0f);
        }
        for (int i = 0; i < 600; i++) world.Step(Dt); // settle into a pile

        foreach (int iters in new[] { 1, 8 })
        {
            world.Solver.Iterations = iters;
            var m = Measure3D(world, frames, iters);
            Console.WriteLine(
                $"[pile3d] iters={iters}: {m.MsPerStep:F3} ms/step " +
                $"pairs={m.Pairs} contacts={m.Contacts} gc0={m.Gc0:F3}/step hash=0x{m.Hash:x}");
        }
        return true;
    }

    /// <summary>3D version of <see cref="Measure"/> (median of 5 windows).</summary>
    private static (double MsPerStep, ulong Hash, int Pairs, int Contacts, double Gc0)
        Measure3D(World3 world, int frames, int warmupIters = 8)
    {
        world.Solver.Iterations = warmupIters;
        for (int i = 0; i < 60; i++) world.Step(Dt); // settle briefly

        const int windows = 5;
        var samples = new double[windows];
        for (int s = 0; s < windows; s++)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < frames; i++) world.Step(Dt);
            sw.Stop();
            samples[s] = sw.Elapsed.TotalMilliseconds / frames;
        }
        Array.Sort(samples);

        long gen0 = GC.CollectionCount(0);
        int pairs = 0, contacts = 0;
        for (int i = 0; i < frames; i++)
        {
            world.Step(Dt);
            pairs += world.ContactCache.Count;
            foreach (var list in world.ContactCache.Values) contacts += list.Count;
        }
        double gc0 = (GC.CollectionCount(0) - gen0) / (double)frames;

        return (samples[windows / 2], HashWorld3(world),
            pairs / frames, contacts / frames, gc0);
    }

    private static ulong HashWorld3(World3 world)
    {
        ulong h = 0xcbf29ce484222325UL;
        foreach (var body in world.Bodies)
        {
            foreach (float v in new[]
            {
                body.Transform.Position.X, body.Transform.Position.Y, body.Transform.Position.Z,
                body.Velocity.X, body.Velocity.Y, body.Velocity.Z,
                body.AngularVelocity.X, body.AngularVelocity.Y, body.AngularVelocity.Z
            })
            {
                h ^= (ulong)BitConverter.SingleToInt32Bits(v);
                h *= 0x100000001b3UL;
            }
        }
        return h;
    }

    private static bool RunStack(int frames)
    {
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        for (int i = 0; i < 10; i++)
        {
            world.AddDynamicBody(
                new Vector2(0.0f, 0.05f + i * 1.02f),
                new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        }
        for (int i = 0; i < 600; i++) world.Step(Dt);
        var m = Measure(world, frames);
        Console.WriteLine(
            $"[stack] {m.MsPerStep:F4} ms/step pairs={m.Pairs} contacts={m.Contacts} " +
            $"gc0={m.Gc0:F4}/step hash=0x{m.Hash:x}");
        // Integrity, not perfection: a 10-high exact stack may lean or
        // topple (real physics); what must never happen is tunneling,
        // explosion, or unsettled jitter.
        bool stable = true;
        for (int i = 1; i <= 10; i++)
        {
            var b = world.Bodies[i];
            if (!(b.Transform.Position.Y > -0.6f) || b.Velocity.Length() >= 1.0f)
                stable = false;
            if (!float.IsFinite(b.Transform.Position.X) || !float.IsFinite(b.Velocity.X))
                stable = false;
        }
        Console.WriteLine($"[stack] stable={stable}");
        return stable;
    }

    private static bool RunJoints(int frames)
    {
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddStaticBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)));
        // Pendulum farm: 20 rods + 10 hinges + 5 sliders + 5 welds.
        for (int i = 0; i < 20; i++)
        {
            int ball = world.AddDynamicBody(
                new Vector2(-9.0f + i, -2.0f), new CircleShape(0.3f), 1.0f);
            world.AddJoint(new DistanceJoint(0, ball, Vector2.Zero, Vector2.Zero,
                2.0f, 0.0f, world.Bodies.Count));
        }
        for (int i = 0; i < 10; i++)
        {
            int bob = world.AddDynamicBody(
                new Vector2(12.0f + i, 1.0f), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
            world.AddJoint(new HingeJoint(0, bob, new Vector2(12.0f + i, 0.0f),
                Vector2.Zero, 1.0f, -1.0f, 0.0f, world.Bodies.Count));
        }
        for (int i = 0; i < 5; i++)
        {
            int slider = world.AddDynamicBody(
                new Vector2(-4.0f + i * 2.0f, 3.0f), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
            world.AddJoint(new PrismaticJoint(0, slider,
                new Vector2(-4.0f + i * 2.0f, 3.0f), Vector2.Zero,
                Vector2.UnitX, 1.0f, -1.0f, 0.0f, world.Bodies.Count));
        }
        for (int i = 0; i < 2; i++)
        {
            int a = world.AddDynamicBody(
                new Vector2(6.0f + i * 3.0f, 5.0f), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
            int b = world.AddDynamicBody(
                new Vector2(7.0f + i * 3.0f, 5.0f), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
            world.AddJoint(new FixedJoint(a, b, new Vector2(1.0f, 0.0f), Vector2.Zero,
                0.0f, world.Bodies.Count));
        }
        var m = Measure(world, frames);
        Console.WriteLine(
            $"[joints] {m.MsPerStep:F4} ms/step bodies={world.Bodies.Count} " +
            $"joints={world.Joints.Count} gc0={m.Gc0:F4}/step hash=0x{m.Hash:x}");
        bool finite = true;
        foreach (var b in world.Bodies)
        {
            if (!float.IsFinite(b.Transform.Position.X) || !float.IsFinite(b.Velocity.X))
                finite = false;
        }
        Console.WriteLine($"[joints] finite={finite}");
        return finite;
    }

    /// <summary>
    /// Island scaling: 8 independent 5-box stacks (8 islands). Runs the
    /// same world serial and parallel, reporting both timings and whether
    /// the trajectories are bit-identical. Best-of-5 medians; machine
    /// noise is ~±40%, so read speedup as a trend, identity as a boolean.
    /// </summary>
    private static bool RunIslands(int frames)
    {
        static World2 Build()
        {
            var world = new World2();
            world.Gravity = new Vector2(0.0f, -9.81f);
            world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(60.0f, 0.5f)));
            for (int s = 0; s < 8; s++)
            {
                float x = -21.0f + s * 6.0f;
                for (int i = 0; i < 5; i++)
                {
                    world.AddDynamicBody(
                        new Vector2(x, 0.05f + i * 1.02f),
                        new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
                }
            }
            return world;
        }

        var serial = Build();
        var ms = Measure(serial, frames);
        var parallel = Build();
        parallel.Solver.Parallel = true;
        var mp = Measure(parallel, frames);
        bool identical = ms.Hash == mp.Hash;
        Console.WriteLine(
            $"[islands] serial={ms.MsPerStep:F4} ms/step parallel={mp.MsPerStep:F4} ms/step " +
            $"speedup={ms.MsPerStep / mp.MsPerStep:F2}x identical={identical} " +
            $"serial=0x{ms.Hash:x} parallel=0x{mp.Hash:x}");
        return identical;
    }
}
