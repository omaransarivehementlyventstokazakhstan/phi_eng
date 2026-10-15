using System.Numerics;
using PhiEng.Dynamics;

namespace PhiEng.Demo;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "bench")
        {
            Bench();
            return;
        }
        Console.WriteLine("Physics Engine Demo - 2D (C#)");
        Console.WriteLine("==============================");
        Console.WriteLine();

        // Create a new physics world
        var world = new World2();

        // Set gravity
        world.Gravity = new Vector2(0.0f, -9.81f);

        // Add a static floor
        int floorId = world.AddStaticBody(
            new Vector2(0.0f, -1.0f),
            new BoxShape2(new Vector2(10.0f, 0.5f))
        );
        Console.WriteLine($"Added floor at y=-1.0 (id: {floorId})");

        // Add a static wall on the left
        int wallId = world.AddStaticBody(
            new Vector2(-5.0f, 0.0f),
            new BoxShape2(new Vector2(0.5f, 5.0f))
        );
        Console.WriteLine($"Added left wall at x=-5.0 (id: {wallId})");

        // Add a static wall on the right
        int wall2Id = world.AddStaticBody(
            new Vector2(5.0f, 0.0f),
            new BoxShape2(new Vector2(0.5f, 5.0f))
        );
        Console.WriteLine($"Added right wall at x=5.0 (id: {wall2Id})");

        // Add some dynamic balls
        Console.WriteLine();
        Console.WriteLine("Dropping balls...");
        for (int i = 0; i < 10; i++)
        {
            float x = -4.0f + (i * 0.8f);
            float y = 5.0f + (i * 0.5f);
            int ballId = world.AddDynamicBody(
                new Vector2(x, y),
                new CircleShape(0.3f),
                1.0f // density
            );
            Console.WriteLine($"  Ball {ballId} at ({x:F1}, {y:F1}) (id: {ballId})");
        }

        // Add a dynamic box
        int boxId = world.AddDynamicBody(
            new Vector2(0.0f, 3.0f),
            new BoxShape2(new Vector2(0.5f, 0.5f)),
            1.0f
        );
        Console.WriteLine($"  Box at (0.0, 3.0) (id: {boxId})");

        // Simulate for 60 frames (1 second at 60 FPS)
        Console.WriteLine();
        Console.WriteLine("Simulating for 1 second (60 frames)...");
        float dt = 1.0f / 60.0f;
        for (int frame = 0; frame < 60; frame++)
        {
            world.Step(dt);

            // Print state every 15 frames
            if (frame % 15 == 0)
            {
                Console.WriteLine($"  Frame {frame}: ");
                foreach (var body in world.Bodies)
                {
                    if (!body.IsStatic)
                    {
                        Console.WriteLine(
                            $"    Body {body.Id}: pos=({body.Transform.Position.X:F2}, {body.Transform.Position.Y:F2}), vel=({body.Velocity.X:F2}, {body.Velocity.Y:F2})"
                        );
                    }
                }
            }
        }

        Console.WriteLine();
        Console.WriteLine("Final state:");
        foreach (var body in world.Bodies)
        {
            Console.WriteLine(
                $"  Body {body.Id}: pos=({body.Transform.Position.X:F2}, {body.Transform.Position.Y:F2}), vel=({body.Velocity.X:F2}, {body.Velocity.Y:F2}), static={body.IsStatic}"
            );
        }

        Console.WriteLine();
        Console.WriteLine("Demo complete!");
    }

    static void Bench()
    {
        const int Bodies = 1500;
        const int Frames = 300;
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(30.0f, 0.5f)));
        for (int i = 0; i < Bodies; i++)
        {
            float x = -14.0f + (i % 30) * 0.95f;
            float y = 2.0f + (i / 30) * 0.75f;
            if (i % 3 == 0)
                world.AddDynamicBody(new Vector2(x, y), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
            else
                world.AddDynamicBody(new Vector2(x, y), new CircleShape(0.3f), 1.0f);
        }
        float dt = 1.0f / 60.0f;
        for (int i = 0; i < 900; i++) world.Step(dt); // settle into a pile

        foreach (int iters in new[] { 1, 8 })
        {
            world.Solver.Iterations = iters;
            var swI = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < Frames; i++) world.Step(dt);
            swI.Stop();
            Console.WriteLine($"iters={iters}: {swI.Elapsed.TotalMilliseconds / Frames:F3} ms/step");
        }
        world.Solver.Iterations = 8;

        long gen0 = GC.CollectionCount(0), gen1 = GC.CollectionCount(1), gen2 = GC.CollectionCount(2);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int totalPairs = 0, totalContacts = 0;
        for (int i = 0; i < Frames; i++)
        {
            world.Step(dt);
            totalPairs += world.ContactCache.Count;
            foreach (var list in world.ContactCache.Values) totalContacts += list.Count;
        }
        sw.Stop();
        Console.WriteLine($"avg pairs/step: {totalPairs / (float)Frames:F0}");
        Console.WriteLine($"avg contacts/step: {totalContacts / (float)Frames:F0}");
        Console.WriteLine($"bodies: {world.Bodies.Count}");
        Console.WriteLine($"frames: {Frames}");
        Console.WriteLine($"total: {sw.Elapsed.TotalMilliseconds:F2} ms");
        Console.WriteLine($"per step: {sw.Elapsed.TotalMilliseconds / Frames:F3} ms");
        Console.WriteLine($"GC gen0/step: {(GC.CollectionCount(0) - gen0) / (float)Frames:F3}");
        Console.WriteLine($"GC gen1/step: {(GC.CollectionCount(1) - gen1) / (float)Frames:F3}");
        Console.WriteLine($"GC gen2/step: {(GC.CollectionCount(2) - gen2) / (float)Frames:F3}");

        // Resting-contact steady state.
        var calm = new World2();
        calm.Gravity = new Vector2(0.0f, -9.81f);
        calm.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        calm.AddDynamicBody(new Vector2(0.0f, 0.2f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        for (int i = 0; i < 600; i++) calm.Step(dt);
        gen0 = GC.CollectionCount(0); gen1 = GC.CollectionCount(1); gen2 = GC.CollectionCount(2);
        sw.Restart();
        for (int i = 0; i < Frames; i++) calm.Step(dt);
        sw.Stop();
        Console.WriteLine($"resting per step: {sw.Elapsed.TotalMilliseconds / Frames:F4} ms");
        Console.WriteLine($"resting GC gen0/step: {(GC.CollectionCount(0) - gen0) / (float)Frames:F4}");
        Console.WriteLine($"resting GC gen1/step: {(GC.CollectionCount(1) - gen1) / (float)Frames:F4}");
        Console.WriteLine($"resting GC gen2/step: {(GC.CollectionCount(2) - gen2) / (float)Frames:F4}");
    }
}
