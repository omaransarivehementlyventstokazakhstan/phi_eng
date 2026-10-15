using System.Numerics;
using PhiEng.Dynamics;
using System.IO;

/// <summary>
/// hello-snapshot: Save/load a 2D world with joints.
/// 
/// Run: dotnet run --project samples/hello-snapshot
/// </summary>
internal static class Program
{
    private const float Dt = 1.0f / 60.0f;
    private const string SavePath = "world_snapshot.json";

    private static void Main()
    {
        var world = BuildWorld();
        Console.WriteLine("Running simulation for 120 steps...");
        for (int i = 0; i < 120; i++) world.Step(Dt);

        Console.WriteLine($"Saving world to {SavePath} (Exact mode)...");
        var json = Snapshot.World2ToJson(world, SnapshotLoadMode.Exact);
        File.WriteAllText(SavePath, json);

        // Modify world
        world.Bodies[1].Velocity = new Vector2(10, 10);
        Console.WriteLine("Modified world (kicked the ball).");

        Console.WriteLine($"Loading world from {SavePath} (Exact mode)...");
        var json2 = File.ReadAllText(SavePath);
        var loaded = Snapshot.World2FromJson(json2, SnapshotLoadMode.Exact);

        Console.WriteLine("Original vs Loaded body positions:");
        for (int i = 0; i < world.Bodies.Count; i++)
        {
            var o = world.Bodies[i].Transform.Position;
            var l = loaded.Bodies[i].Transform.Position;
            Console.WriteLine($"  Body {i}: orig=({o.X:F3},{o.Y:F3}) loaded=({l.X:F3},{l.Y:F3}) match={o == l}");
        }
        Console.WriteLine("Done. Positions should match exactly (bit-identical).");
    }

    private static World2 BuildWorld()
    {
        var world = new World2();
        world.Gravity = new Vector2(0, -9.81f);
        world.AddStaticBody(new Vector2(0, -1), new BoxShape2(new Vector2(10, 0.5f)));
        world.AddDynamicBody(new Vector2(0, 5), new CircleShape(0.5f), 1.0f);
        return world;
    }
}