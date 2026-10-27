using System.Numerics;

namespace PhiEng.Electromagnetics;

/// <summary>
/// How the electromagnetic force is computed each step. Discriminants are
/// part of the snapshot file format; never reorder. Mirrors Rust
/// <c>phieng_electromagnetics::ElectroMode</c>.
/// </summary>
public enum ElectroMode : uint
{
    /// <summary>No electromagnetic force.</summary>
    Off = 0,
    /// <summary>Direct Coulomb sum plus the uniform external fields.</summary>
    Pairwise = 1,
    /// <summary>A grid field (see <see cref="PoissonGrid2"/>).</summary>
    Field = 2,
}

/// <summary>The boundary condition of a field grid.</summary>
public enum FieldBoundary : uint
{
    /// <summary>The potential is zero on the boundary.</summary>
    Dirichlet = 0,
    /// <summary>The normal derivative is zero.</summary>
    Neumann = 1,
    /// <summary>The grid wraps around.</summary>
    Periodic = 2,
}

/// <summary>The 2D field-grid configuration.</summary>
public struct FieldConfig2
{
    /// <summary>Cells along X.</summary>
    public uint Nx { get; set; }
    /// <summary>Cells along Y.</summary>
    public uint Ny { get; set; }
    /// <summary>Cell size.</summary>
    public float Dx { get; set; }
    /// <summary>Grid origin.</summary>
    public Vector2 Origin { get; set; }
    /// <summary>Fixed SOR iterations.</summary>
    public uint Iterations { get; set; }
    /// <summary>SOR over-relaxation in (0, 2).</summary>
    public float Omega { get; set; }
    /// <summary>The boundary condition.</summary>
    public FieldBoundary Boundary { get; set; }
    /// <summary>Relative permittivity of the medium.</summary>
    public float EpsR { get; set; }
    /// <summary>Whether the field is stored in a snapshot.</summary>
    public bool SnapshotField { get; set; }

    /// <summary>A grid with the engine defaults (128x128, dx 0.1).</summary>
    public FieldConfig2()
    {
        Nx = 128;
        Ny = 128;
        Dx = 0.1f;
        Origin = Vector2.Zero;
        Iterations = 40;
        Omega = 1.8f;
        Boundary = FieldBoundary.Dirichlet;
        EpsR = 1.0f;
        SnapshotField = true;
    }

    /// <summary>The engine default (128x128, dx 0.1).</summary>
    public static FieldConfig2 Default => new();

    /// <summary>Number of cells.</summary>
    public readonly int Cells => checked((int)((long)Nx * Ny));
}

/// <summary>The 3D field-grid configuration.</summary>
public struct FieldConfig3
{
    /// <summary>Cells along X.</summary>
    public uint Nx { get; set; }
    /// <summary>Cells along Y.</summary>
    public uint Ny { get; set; }
    /// <summary>Cells along Z.</summary>
    public uint Nz { get; set; }
    /// <summary>Cell size.</summary>
    public float Dx { get; set; }
    /// <summary>Grid origin.</summary>
    public Vector3 Origin { get; set; }
    /// <summary>Fixed SOR iterations.</summary>
    public uint Iterations { get; set; }
    /// <summary>SOR over-relaxation in (0, 2).</summary>
    public float Omega { get; set; }
    /// <summary>The boundary condition.</summary>
    public FieldBoundary Boundary { get; set; }
    /// <summary>Relative permittivity of the medium.</summary>
    public float EpsR { get; set; }
    /// <summary>Whether the field is stored in a snapshot.</summary>
    public bool SnapshotField { get; set; }

    /// <summary>A grid with the engine defaults (32^3, dx 0.25).</summary>
    public FieldConfig3()
    {
        Nx = 32;
        Ny = 32;
        Nz = 32;
        Dx = 0.25f;
        Origin = Vector3.Zero;
        Iterations = 40;
        Omega = 1.8f;
        Boundary = FieldBoundary.Dirichlet;
        EpsR = 1.0f;
        SnapshotField = true;
    }

    /// <summary>The engine default (32^3, dx 0.25).</summary>
    public static FieldConfig3 Default => new();

    /// <summary>Number of cells.</summary>
    public readonly int Cells => checked((int)((long)Nx * Ny * Nz));
}

/// <summary>The 2D electromagnetic configuration. Mirrors Rust <c>ElectroConfig2</c>.</summary>
public struct ElectroConfig2
{
    /// <summary>Master switch.</summary>
    public bool Enabled { get; set; }
    /// <summary>The force model.</summary>
    public ElectroMode Mode { get; set; }
    /// <summary>Coulomb's constant.</summary>
    public float CoulombK { get; set; }
    /// <summary>Softening length.</summary>
    public float Softening { get; set; }
    /// <summary>The uniform external electric field.</summary>
    public Vector2 UniformE { get; set; }
    /// <summary>The uniform external magnetic field (out of plane).</summary>
    public float UniformB { get; set; }
    /// <summary>Relative permittivity of the medium.</summary>
    public float Permittivity { get; set; }
    /// <summary>Relative permeability of the medium.</summary>
    public float Permeability { get; set; }
    /// <summary>The acceleration cap.</summary>
    public float MaxAcceleration { get; set; }
    /// <summary>The field grid.</summary>
    public FieldConfig2 Field { get; set; }

    /// <summary>A configuration with the engine defaults (off).</summary>
    public ElectroConfig2()
    {
        Enabled = false;
        Mode = ElectroMode.Off;
        CoulombK = ElectroConstants.CoulombK;
        Softening = 0.0f;
        UniformE = Vector2.Zero;
        UniformB = 0.0f;
        Permittivity = 1.0f;
        Permeability = 1.0f;
        MaxAcceleration = ElectroConstants.MaxAcceleration;
        Field = new FieldConfig2();
    }

    /// <summary>The engine default (off).</summary>
    public static ElectroConfig2 Default => new();

    /// <summary>True when every field holds its default.</summary>
    public readonly bool IsDefault()
    {
        var d = Default;
        return !Enabled
            && Mode == d.Mode
            && CoulombK == d.CoulombK
            && Softening == d.Softening
            && UniformE == d.UniformE
            && UniformB == d.UniformB
            && Permittivity == d.Permittivity
            && Permeability == d.Permeability
            && MaxAcceleration == d.MaxAcceleration
            && Field.Equals(d.Field);
    }

    /// <summary>Validate every field. A disabled config is inert.</summary>
    public readonly bool Validate()
    {
        if (!Enabled) return true;
        if (!float.IsFinite(CoulombK) || CoulombK <= 0.0f) return false;
        if (!float.IsFinite(Softening) || Softening < 0.0f) return false;
        if (!IsFinite(UniformE) || !float.IsFinite(UniformB)) return false;
        if (!float.IsFinite(Permittivity) || Permittivity < ElectroConstants.MinMaterial) return false;
        if (!float.IsFinite(Permeability) || Permeability < ElectroConstants.MinMaterial) return false;
        if (!float.IsFinite(MaxAcceleration) || MaxAcceleration <= 0.0f) return false;
        if (Mode == ElectroMode.Field)
        {
            if (Field.Nx == 0 || Field.Ny == 0) return false;
            if (!float.IsFinite(Field.Dx) || Field.Dx <= 0.0f || !IsFinite(Field.Origin)) return false;
            if ((long)Field.Nx * Field.Ny > ElectroConstants.MaxFieldCells) return false;
            if (Field.Iterations == 0) return false;
            if (!float.IsFinite(Field.Omega) || Field.Omega <= 0.0f || Field.Omega >= 2.0f) return false;
            if (!float.IsFinite(Field.EpsR) || Field.EpsR < ElectroConstants.MinMaterial) return false;
        }
        return true;
    }

    private static bool IsFinite(Vector2 v) => float.IsFinite(v.X) && float.IsFinite(v.Y);
}

/// <summary>The 3D electromagnetic configuration. Mirrors Rust <c>ElectroConfig3</c>.</summary>
public struct ElectroConfig3
{
    /// <summary>Master switch.</summary>
    public bool Enabled { get; set; }
    /// <summary>The force model.</summary>
    public ElectroMode Mode { get; set; }
    /// <summary>Coulomb's constant.</summary>
    public float CoulombK { get; set; }
    /// <summary>Softening length.</summary>
    public float Softening { get; set; }
    /// <summary>The uniform external electric field.</summary>
    public Vector3 UniformE { get; set; }
    /// <summary>The uniform external magnetic field.</summary>
    public Vector3 UniformB { get; set; }
    /// <summary>Relative permittivity of the medium.</summary>
    public float Permittivity { get; set; }
    /// <summary>Relative permeability of the medium.</summary>
    public float Permeability { get; set; }
    /// <summary>The acceleration cap.</summary>
    public float MaxAcceleration { get; set; }
    /// <summary>The field grid.</summary>
    public FieldConfig3 Field { get; set; }

    /// <summary>A configuration with the engine defaults (off).</summary>
    public ElectroConfig3()
    {
        Enabled = false;
        Mode = ElectroMode.Off;
        CoulombK = ElectroConstants.CoulombK;
        Softening = 0.0f;
        UniformE = Vector3.Zero;
        UniformB = Vector3.Zero;
        Permittivity = 1.0f;
        Permeability = 1.0f;
        MaxAcceleration = ElectroConstants.MaxAcceleration;
        Field = new FieldConfig3();
    }

    /// <summary>The engine default (off).</summary>
    public static ElectroConfig3 Default => new();

    /// <summary>True when every field holds its default.</summary>
    public readonly bool IsDefault()
    {
        var d = Default;
        return !Enabled
            && Mode == d.Mode
            && CoulombK == d.CoulombK
            && Softening == d.Softening
            && UniformE == d.UniformE
            && UniformB == d.UniformB
            && Permittivity == d.Permittivity
            && Permeability == d.Permeability
            && MaxAcceleration == d.MaxAcceleration
            && Field.Equals(d.Field);
    }

    /// <summary>Validate every field. A disabled config is inert.</summary>
    public readonly bool Validate()
    {
        if (!Enabled) return true;
        if (!float.IsFinite(CoulombK) || CoulombK <= 0.0f) return false;
        if (!float.IsFinite(Softening) || Softening < 0.0f) return false;
        if (!IsFinite(UniformE) || !IsFinite(UniformB)) return false;
        if (!float.IsFinite(Permittivity) || Permittivity < ElectroConstants.MinMaterial) return false;
        if (!float.IsFinite(Permeability) || Permeability < ElectroConstants.MinMaterial) return false;
        if (!float.IsFinite(MaxAcceleration) || MaxAcceleration <= 0.0f) return false;
        if (Mode == ElectroMode.Field)
        {
            if (Field.Nx == 0 || Field.Ny == 0 || Field.Nz == 0) return false;
            if (!float.IsFinite(Field.Dx) || Field.Dx <= 0.0f || !IsFinite(Field.Origin)) return false;
            if ((long)Field.Nx * Field.Ny * Field.Nz > ElectroConstants.MaxFieldCells) return false;
            if (Field.Iterations == 0) return false;
            if (!float.IsFinite(Field.Omega) || Field.Omega <= 0.0f || Field.Omega >= 2.0f) return false;
            if (!float.IsFinite(Field.EpsR) || Field.EpsR < ElectroConstants.MinMaterial) return false;
        }
        return true;
    }

    private static bool IsFinite(Vector3 v) =>
        float.IsFinite(v.X) && float.IsFinite(v.Y) && float.IsFinite(v.Z);
}

/// <summary>The 2D per-body electromagnetic state. Mirrors Rust <c>ElectroState2</c>.</summary>
public struct ElectroState2
{
    /// <summary>The net charge.</summary>
    public float Charge { get; set; }
    /// <summary>The coupling multiplier.</summary>
    public float ChargeScale { get; set; }
    /// <summary>Reserved conductivity.</summary>
    public float Conductivity { get; set; }
    /// <summary>Relative permittivity.</summary>
    public float Permittivity { get; set; }
    /// <summary>Relative permeability.</summary>
    public float Permeability { get; set; }
    /// <summary>The out-of-plane magnetic moment.</summary>
    public float MagneticMoment { get; set; }
    /// <summary>The capacitance to ground.</summary>
    public float Capacitance { get; set; }
    /// <summary>The resistance to ground.</summary>
    public float Resistance { get; set; }
    /// <summary>Whether the body is a conductor.</summary>
    public bool IsConductor { get; set; }
    /// <summary>Whether the conductor is grounded.</summary>
    public bool IsGrounded { get; set; }

    /// <summary>A state with the engine defaults (uncharged, unit coupling).</summary>
    public ElectroState2()
    {
        Charge = 0.0f;
        ChargeScale = 1.0f;
        Conductivity = 0.0f;
        Permittivity = 1.0f;
        Permeability = 1.0f;
        MagneticMoment = 0.0f;
        Capacitance = 1.0f;
        Resistance = 0.0f;
        IsConductor = false;
        IsGrounded = false;
    }

    /// <summary>The engine default (uncharged, unit coupling).</summary>
    public static ElectroState2 Default => new();

    /// <summary>True when every field holds its default.</summary>
    public readonly bool IsDefault() => this.Equals(Default);

    /// <summary>Clamp to the engine's safe ranges, dropping non-finite values.</summary>
    public readonly ElectroState2 Sanitized()
    {
        var s = this;
        if (!float.IsFinite(s.Charge)) s.Charge = 0.0f;
        s.Charge = System.Math.Clamp(s.Charge, -ElectroConstants.MaxCharge, ElectroConstants.MaxCharge);
        if (!float.IsFinite(s.ChargeScale)) s.ChargeScale = 1.0f;
        if (!float.IsFinite(s.Conductivity) || s.Conductivity < 0.0f) s.Conductivity = 0.0f;
        if (!float.IsFinite(s.Permittivity) || s.Permittivity < ElectroConstants.MinMaterial) s.Permittivity = 1.0f;
        if (!float.IsFinite(s.Permeability) || s.Permeability < ElectroConstants.MinMaterial) s.Permeability = 1.0f;
        if (!float.IsFinite(s.MagneticMoment)) s.MagneticMoment = 0.0f;
        if (!float.IsFinite(s.Capacitance) || s.Capacitance <= 0.0f) s.Capacitance = 1.0f;
        if (!float.IsFinite(s.Resistance) || s.Resistance < 0.0f) s.Resistance = 0.0f;
        return s;
    }
}

/// <summary>The 3D per-body electromagnetic state. Mirrors Rust <c>ElectroState3</c>.</summary>
public struct ElectroState3
{
    /// <summary>The net charge.</summary>
    public float Charge { get; set; }
    /// <summary>The coupling multiplier.</summary>
    public float ChargeScale { get; set; }
    /// <summary>Reserved conductivity.</summary>
    public float Conductivity { get; set; }
    /// <summary>Relative permittivity.</summary>
    public float Permittivity { get; set; }
    /// <summary>Relative permeability.</summary>
    public float Permeability { get; set; }
    /// <summary>The magnetic dipole moment.</summary>
    public Vector3 MagneticMoment { get; set; }
    /// <summary>The capacitance to ground.</summary>
    public float Capacitance { get; set; }
    /// <summary>The resistance to ground.</summary>
    public float Resistance { get; set; }
    /// <summary>Whether the body is a conductor.</summary>
    public bool IsConductor { get; set; }
    /// <summary>Whether the conductor is grounded.</summary>
    public bool IsGrounded { get; set; }

    /// <summary>A state with the engine defaults (uncharged, unit coupling).</summary>
    public ElectroState3()
    {
        Charge = 0.0f;
        ChargeScale = 1.0f;
        Conductivity = 0.0f;
        Permittivity = 1.0f;
        Permeability = 1.0f;
        MagneticMoment = Vector3.Zero;
        Capacitance = 1.0f;
        Resistance = 0.0f;
        IsConductor = false;
        IsGrounded = false;
    }

    /// <summary>The engine default (uncharged, unit coupling).</summary>
    public static ElectroState3 Default => new();

    /// <summary>True when every field holds its default.</summary>
    public readonly bool IsDefault() => this.Equals(Default);

    /// <summary>Clamp to the engine's safe ranges, dropping non-finite values.</summary>
    public readonly ElectroState3 Sanitized()
    {
        var s = this;
        if (!float.IsFinite(s.Charge)) s.Charge = 0.0f;
        s.Charge = System.Math.Clamp(s.Charge, -ElectroConstants.MaxCharge, ElectroConstants.MaxCharge);
        if (!float.IsFinite(s.ChargeScale)) s.ChargeScale = 1.0f;
        if (!float.IsFinite(s.Conductivity) || s.Conductivity < 0.0f) s.Conductivity = 0.0f;
        if (!float.IsFinite(s.Permittivity) || s.Permittivity < ElectroConstants.MinMaterial) s.Permittivity = 1.0f;
        if (!float.IsFinite(s.Permeability) || s.Permeability < ElectroConstants.MinMaterial) s.Permeability = 1.0f;
        if (!float.IsFinite(s.MagneticMoment.X) || !float.IsFinite(s.MagneticMoment.Y) || !float.IsFinite(s.MagneticMoment.Z))
            s.MagneticMoment = Vector3.Zero;
        if (!float.IsFinite(s.Capacitance) || s.Capacitance <= 0.0f) s.Capacitance = 1.0f;
        if (!float.IsFinite(s.Resistance) || s.Resistance < 0.0f) s.Resistance = 0.0f;
        return s;
    }
}



