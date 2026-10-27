using System.Numerics;

namespace PhiEng.Electromagnetics;

/// <summary>
/// The Lorentz acceleration in uniform fields and the torque on a magnetic
/// dipole. Mirrors Rust <c>phieng-electromagnetics::magnetostatics</c>.
/// </summary>
public static class Magnetostatics
{
    /// <summary>
    /// The 2D acceleration of a charge: <c>a = (q/m) (E + v x B)</c> with
    /// <c>B = Bz z</c>, so <c>v x B = (vy Bz, -vx Bz)</c>.
    /// </summary>
    public static Vector2 LorentzAccel2(Vector2 v, float qOverM, Vector2 e, float bz)
    {
        var vxb = new Vector2(v.Y * bz, -v.X * bz);
        return (e + vxb) * qOverM;
    }

    /// <summary>The 3D counterpart of <see cref="LorentzAccel2"/>.</summary>
    public static Vector3 LorentzAccel3(Vector3 v, float qOverM, Vector3 e, Vector3 b)
    {
        return (e + Vector3.Cross(v, b)) * qOverM;
    }

    /// <summary>The torque a uniform field applies to a dipole: <c>m x B</c>.</summary>
    public static Vector3 DipoleTorque3(Vector3 moment, Vector3 b)
    {
        return Vector3.Cross(moment, b);
    }
}
