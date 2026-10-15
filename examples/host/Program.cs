using System.Numerics;
using PhiEng.Dynamics;

/// <summary>
/// PhiEng.Host: A minimal host application for CI smoke testing.
/// 
/// Usage: dotnet run --project examples/host -- --smoke --steps 600
/// </summary>
internal static class Program
{
    private static void Main(string[] args)
    {
        bool smoke = args.Contains("--smoke");
        int steps = 600;
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == "--steps" && int.TryParse(args[i + 1], out int s))
                steps = s;
        }

        var world = new World2();
        world.Gravity = new Vector2(0, -9.81f);
        world.AddStaticBody(new Vector2(0, -1), new BoxShape2(new Vector2(10, 0.5f)));
        int ball = world.AddDynamicBody(new Vector2(0, 5), new CircleShape(0.5f), 1.0f);

        float dt = 1.0f / 60.0f;
        for (int i = 0; i < steps; i++)
            world.Step(dt);

        var pos = world.Bodies[ball].Transform.Position;
        Console.WriteLine($"host-smoke: steps={steps} final_pos=({pos.X:F3}, {pos.Y:F3})");

        if (smoke)
        {
            // Basic sanity checks
            if (!float.IsFinite(pos.X) || !float.IsFinite(pos.Y))
                throw new Exception("host-smoke: NaN position");
            if (pos.Y < -0.1f)
                throw new Exception("host-smoke: body fell through floor");
        }
    }
}