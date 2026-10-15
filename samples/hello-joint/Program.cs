using System.Numerics;
using PhiEng.Dynamics;

/// <summary>
/// hello-joint: A 5-minute 2D joints demo (distance rod + hinge).
/// 
/// Run: dotnet run --project samples/hello-joint
/// </summary>
internal static class Program
{
    private const float Dt = 1.0f / 60.0f;

    private static void Main()
    {
        var world = new World2();
        world.Gravity = new Vector2(0, -9.81f);

        // Static anchor at origin
        int anchor = world.AddStaticBody(Vector2.Zero, new BoxShape2(new Vector2(0.5f, 0.5f)));

        // Pendulum bob
        int bob = world.AddDynamicBody(new Vector2(2, 0), new CircleShape(0.3f), 1.0f);

        // Distance joint (rod) from anchor to bob, rest length 2
        world.AddJoint(new DistanceJoint(anchor, bob, Vector2.Zero, Vector2.Zero, 2.0f, 0.0f, world.Bodies.Count));

        Console.WriteLine("2D Joints: Pendulum on a rigid rod...");
        for (int step = 0; step < 300; step++)
        {
            world.Step(Dt);
            var b = world.Bodies[bob];
            if (step % 60 == 0)
            {
                Console.WriteLine($"  t={step*Dt:F2}s  pos=({b.Transform.Position.X:F3}, {b.Transform.Position.Y:F3})  dist={b.Transform.Position.Length():F3}");
            }
        }
        Console.WriteLine("Done. Bob should swing and settle at rest length.");
    }
}