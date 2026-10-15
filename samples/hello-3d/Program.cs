using System.Numerics;
using PhiEng.Dynamics;

/// <summary>
/// hello-3d: A 5-minute 3D physics demo.
/// 
/// Run: dotnet run --project samples/hello-3d
/// </summary>
internal static class Program
{
    private const float Dt = 1.0f / 60.0f;

    private static void Main()
    {
        var world = new World3();
        world.Gravity = new Vector3(0, -9.81f, 0);

        // Static floor
        world.AddStaticBody(new Vector3(0, -1, 0), new BoxShape3(new Vector3(10, 0.5f, 10)));

        // Drop a sphere from height 5
        int sphere = world.AddDynamicBody(new Vector3(0, 5, 0), new SphereShape3(0.5f), 1.0f);

        Console.WriteLine("3D Physics: Dropping a sphere onto a floor...");
        for (int step = 0; step < 300; step++)
        {
            world.Step(Dt);
            var b = world.Bodies[sphere];
            if (step % 60 == 0)
            {
                Console.WriteLine($"  t={step*Dt:F2}s  pos=({b.Transform.Position.X:F3}, {b.Transform.Position.Y:F3}, {b.Transform.Position.Z:F3})  vel=({b.Velocity.X:F3}, {b.Velocity.Y:F3}, {b.Velocity.Z:F3})");
            }
        }
        Console.WriteLine("Done. Sphere should rest on floor.");
    }
}