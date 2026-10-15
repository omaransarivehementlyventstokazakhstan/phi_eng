using System.Numerics;
using PhiEng.Native;

/// <summary>
/// hello-native: Use the native Rust core via FFI bridge.
/// 
/// Run: dotnet run --project samples/hello-native
/// </summary>
internal static class Program
{
    private const float Dt = 1.0f / 60.0f;

    private static void Main()
    {
        using var world = new PhiNativeWorld();
        world.SetGravity(new Vector2(0, -9.81f));

        // Static floor
        int floor = world.AddStaticBody(new Vector2(0, -1), new Vector2(10, 0.5f));
        Console.WriteLine($"Floor body id: {floor}");

        // Drop a circle from height 5
        int ball = world.AddDynamicBall(new Vector2(0, 5), 0.5f, 1.0f);
        Console.WriteLine($"Ball body id: {ball}");

        Console.WriteLine("Native FFI: Dropping a ball onto a floor...");
        for (int step = 0; step < 300; step++)
        {
            world.Step(Dt);
            if (step % 60 == 0)
            {
                var (pos, rot) = world.GetBodyTransform(ball);
                Console.WriteLine($"  t={step*Dt:F2}s  pos=({pos.X:F3}, {pos.Y:F3})  rot={rot:F3}");
            }
        }
        Console.WriteLine("Done. Ball should rest on floor via native core.");
    }
}