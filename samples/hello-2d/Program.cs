using System.Numerics;
using PhiEng.Dynamics;

/// <summary>
/// hello-2d: A 5-minute 2D physics demo.
/// 
/// Run: dotnet run --project samples/hello-2d
/// </summary>
internal static class Program
{
    private const float Dt = 1.0f / 60.0f;

    private static void Main()
    {
        var world = new World2();
        world.Gravity = new Vector2(0, -9.81f);

        // Static floor
        world.AddStaticBody(new Vector2(0, -1), new BoxShape2(new Vector2(10, 0.5f)));

        // Drop a box from height 5
        int box = world.AddDynamicBody(new Vector2(0, 5), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);

        Console.WriteLine("2D Physics: Dropping a box onto a floor...");
        for (int step = 0; step < 300; step++)
        {
            world.Step(Dt);
            var b = world.Bodies[box];
            if (step % 60 == 0)
            {
                Console.WriteLine($"  t={step*Dt:F2}s  pos=({b.Transform.Position.X:F3}, {b.Transform.Position.Y:F3})  vel=({b.Velocity.X:F3}, {b.Velocity.Y:F3})");
            }
        }
        Console.WriteLine("Done. Box should rest on floor.");
    }
}