using System.Globalization;
using System.Numerics;
using System.Text;

namespace PhiEng.Native;

/// <summary>
/// Serialises a query shape into the JSON the native overlap query
/// accepts.
/// </summary>
/// <remarks>
/// The native side parses this with its own strict reader, so the
/// numbers are written round-trippably (<c>"R"</c>) in the invariant
/// culture; a comma-decimal culture would otherwise produce JSON the
/// bridge rejects. Shapes the bridge does not know are refused here
/// rather than sent and rejected silently.
/// </remarks>
internal static class ShapeJson
{
    private static string F(float v) => v.ToString("R", CultureInfo.InvariantCulture);

    /// <summary>The 2D shape as bridge JSON.</summary>
    public static string From(Dynamics.Shape2 shape)
    {
        switch (shape)
        {
            case Dynamics.CircleShape c:
                return $"{{\"type\":\"circle\",\"radius\":{F(c.Radius)}}}";
            case Dynamics.BoxShape2 b:
                return $"{{\"type\":\"box\",\"halfExtents\":[{F(b.HalfExtents.X)},{F(b.HalfExtents.Y)}]}}";
            case Dynamics.PolygonShape p:
            {
                var sb = new StringBuilder("{\"type\":\"polygon\",\"vertices\":[");
                for (int i = 0; i < p.Vertices.Count; i++)
                {
                    if (i > 0) sb.Append(',');
                    sb.Append('[').Append(F(p.Vertices[i].X)).Append(',').Append(F(p.Vertices[i].Y)).Append(']');
                }
                return sb.Append("]}").ToString();
            }
            default:
                throw new ArgumentException($"Unsupported 2D query shape: {shape.GetType().Name}.");
        }
    }

    /// <summary>The 3D shape as bridge JSON.</summary>
    public static string From(Dynamics.Shape3 shape)
    {
        switch (shape)
        {
            case Dynamics.SphereShape3 s:
                return $"{{\"type\":\"sphere\",\"radius\":{F(s.Radius)}}}";
            case Dynamics.BoxShape3 b:
                return $"{{\"type\":\"box\",\"halfExtents\":[{F(b.HalfExtents.X)},{F(b.HalfExtents.Y)},{F(b.HalfExtents.Z)}]}}";
            case Dynamics.CapsuleShape3 c:
                return $"{{\"type\":\"capsule\",\"radius\":{F(c.Radius)},\"halfLength\":{F(c.HalfLength)}}}";
            case Dynamics.ConvexHullShape3 h:
            {
                var sb = new StringBuilder("{\"type\":\"hull\",\"vertices\":[");
                for (int i = 0; i < h.Vertices.Count; i++)
                {
                    if (i > 0) sb.Append(',');
                    var v = h.Vertices[i];
                    sb.Append('[').Append(F(v.X)).Append(',').Append(F(v.Y)).Append(',').Append(F(v.Z)).Append(']');
                }
                return sb.Append("]}").ToString();
            }
            default:
                throw new ArgumentException($"Unsupported 3D query shape: {shape.GetType().Name}.");
        }
    }
}
