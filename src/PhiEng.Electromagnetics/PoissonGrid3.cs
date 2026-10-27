using System.Numerics;

namespace PhiEng.Electromagnetics;

/// <summary>
/// The 3D electrostatic grid: red-black SOR for
/// <c>nabla^2 phi = -rho / (eps0 eps_r)</c>. Mirrors Rust
/// <c>PoissonGrid3</c>; the 3D grid and the pairwise solver share the
/// 1/r^2 point-charge law.
/// </summary>
public sealed class PoissonGrid3
{
    /// <summary>The grid configuration.</summary>
    public FieldConfig3 Config { get; }
    /// <summary>The Coulomb constant the grid was solved with.</summary>
    public float CoulombK { get; }
    /// <summary>The solved potential, row-major <c>(z, y, x)</c>.</summary>
    public float[] Phi { get; private set; }
    /// <summary>The deposited charge density, row-major <c>(z, y, x)</c>.</summary>
    public float[] Rho { get; }

    /// <summary>Build a grid, validating its configuration.</summary>
    public PoissonGrid3(FieldConfig3 config, float coulombK)
    {
        if (config.Nx == 0 || config.Ny == 0 || config.Nz == 0)
            throw new ArgumentException("empty grid", nameof(config));
        if (!float.IsFinite(config.Dx) || config.Dx <= 0.0f || !IsFinite(config.Origin))
            throw new ArgumentException("bad grid spacing", nameof(config));
        int cells = checked((int)((long)config.Nx * config.Ny * config.Nz));
        if (cells > ElectroConstants.MaxFieldCells) throw new ArgumentException("grid too large", nameof(config));
        if (config.Iterations == 0 || !float.IsFinite(config.Omega) || config.Omega <= 0.0f || config.Omega >= 2.0f)
            throw new ArgumentException("bad solver tuning", nameof(config));
        if (!float.IsFinite(config.EpsR) || config.EpsR < ElectroConstants.MinMaterial)
            throw new ArgumentException("bad permittivity", nameof(config));
        if (!float.IsFinite(coulombK) || coulombK <= 0.0f) throw new ArgumentException("bad k", nameof(coulombK));
        Config = config;
        CoulombK = coulombK;
        Phi = new float[cells];
        Rho = new float[cells];
    }

    /// <summary>The number of cells.</summary>
    public int Cells => Phi.Length;

    /// <summary>Replace the potential (snapshot loader).</summary>
    public void SetPhi(float[] phi)
    {
        if (phi.Length != Phi.Length) throw new ArgumentException("field length mismatch", nameof(phi));
        Phi = phi;
    }

    /// <summary>Clear the deposited sources.</summary>
    public void ClearSources()
    {
        Array.Clear(Rho);
    }

    /// <summary>Deposit a point charge with trilinear weights.</summary>
    public void DepositCharge(Vector3 pos, float charge)
    {
        if (!IsFinite(pos) || !float.IsFinite(charge) || charge == 0.0f) return;
        float fx = (pos.X - Config.Origin.X) / Config.Dx;
        float fy = (pos.Y - Config.Origin.Y) / Config.Dx;
        float fz = (pos.Z - Config.Origin.Z) / Config.Dx;
        if (!float.IsFinite(fx) || !float.IsFinite(fy) || !float.IsFinite(fz)) return;
        float ix = MathF.Floor(fx);
        float iy = MathF.Floor(fy);
        float iz = MathF.Floor(fz);
        float tx = fx - ix;
        float ty = fy - iy;
        float tz = fz - iz;
        long i0 = (long)ix;
        long j0 = (long)iy;
        long k0 = (long)iz;
        long nx = Config.Nx;
        long ny = Config.Ny;
        long nz = Config.Nz;
        float density = charge / (Config.Dx * Config.Dx * Config.Dx);
        for (int dz = 0; dz < 2; dz++)
        {
            float wz = dz == 0 ? 1.0f - tz : tz;
            for (int dy = 0; dy < 2; dy++)
            {
                float wy = dy == 0 ? 1.0f - ty : ty;
                for (int dx = 0; dx < 2; dx++)
                {
                    float wx = dx == 0 ? 1.0f - tx : tx;
                    float w = wx * wy * wz;
                    if (w == 0.0f) continue;
                    long i = WrapOrClamp(i0 + dx, nx, Config.Boundary);
                    long j = WrapOrClamp(j0 + dy, ny, Config.Boundary);
                    long k = WrapOrClamp(k0 + dz, nz, Config.Boundary);
                    Rho[(k * ny + j) * nx + i] += density * w;
                }
            }
        }
    }

    /// <summary>Solve with red-black SOR. Deterministic iteration order.</summary>
    public void Solve()
    {
        int nx = (int)Config.Nx;
        int ny = (int)Config.Ny;
        int nz = (int)Config.Nz;
        float dx2 = Config.Dx * Config.Dx;
        float scale = dx2 / (ElectroConstants.Eps0(CoulombK) * Config.EpsR);
        for (uint it = 0; it < Config.Iterations; it++)
        {
            for (int color = 0; color < 2; color++)
            {
                for (int k = 0; k < nz; k++)
                {
                    for (int j = 0; j < ny; j++)
                    {
                        for (int i = 0; i < nx; i++)
                        {
                            if ((i + j + k) % 2 != color) continue;
                            int idx = (k * ny + j) * nx + i;
                            float left = PhiAt(i - 1, j, k);
                            float right = PhiAt(i + 1, j, k);
                            float down = PhiAt(i, j - 1, k);
                            float up = PhiAt(i, j + 1, k);
                            float back = PhiAt(i, j, k - 1);
                            float front = PhiAt(i, j, k + 1);
                            float target = (left + right + down + up + back + front + scale * Rho[idx]) / 6.0f;
                            float old = Phi[idx];
                            Phi[idx] = old + Config.Omega * (target - old);
                        }
                    }
                }
            }
        }
    }

    /// <summary>The potential at a world point (trilinear).</summary>
    public float PotentialAt(Vector3 pos)
    {
        var (i0, j0, k0, tx, ty, tz) = Locate(pos);
        float acc = 0.0f;
        for (int dz = 0; dz < 2; dz++)
        {
            float wz = dz == 0 ? 1.0f - tz : tz;
            for (int dy = 0; dy < 2; dy++)
            {
                float wy = dy == 0 ? 1.0f - ty : ty;
                for (int dx = 0; dx < 2; dx++)
                {
                    float wx = dx == 0 ? 1.0f - tx : tx;
                    float w = wx * wy * wz;
                    if (w == 0.0f) continue;
                    acc += w * CellPhi(i0 + dx, j0 + dy, k0 + dz);
                }
            }
        }
        return acc;
    }

    /// <summary>The electric field at a world point: -grad(phi), trilinear.</summary>
    public Vector3 FieldAt(Vector3 pos)
    {
        var (i0, j0, k0, tx, ty, tz) = Locate(pos);
        Vector3 acc = Vector3.Zero;
        for (int dz = 0; dz < 2; dz++)
        {
            float wz = dz == 0 ? 1.0f - tz : tz;
            for (int dy = 0; dy < 2; dy++)
            {
                float wy = dy == 0 ? 1.0f - ty : ty;
                for (int dx = 0; dx < 2; dx++)
                {
                    float wx = dx == 0 ? 1.0f - tx : tx;
                    float w = wx * wy * wz;
                    if (w == 0.0f) continue;
                    acc += w * CellField(i0 + dx, j0 + dy, k0 + dz);
                }
            }
        }
        return acc;
    }

    /// <summary>The electrostatic energy <c>0.5 sum(rho phi) dx^3</c>.</summary>
    public float Energy()
    {
        float sum = 0.0f;
        for (int i = 0; i < Rho.Length; i++) sum += Rho[i] * Phi[i];
        return 0.5f * sum * Config.Dx * Config.Dx * Config.Dx;
    }

    private (int, int, int, float, float, float) Locate(Vector3 pos)
    {
        int nx = (int)Config.Nx;
        int ny = (int)Config.Ny;
        int nz = (int)Config.Nz;
        float fx = (pos.X - Config.Origin.X) / Config.Dx;
        float fy = (pos.Y - Config.Origin.Y) / Config.Dx;
        float fz = (pos.Z - Config.Origin.Z) / Config.Dx;
        if (!float.IsFinite(fx) || !float.IsFinite(fy) || !float.IsFinite(fz))
            return (0, 0, 0, 0.0f, 0.0f, 0.0f);
        float cx = System.Math.Clamp(fx, 0.0f, nx - 1);
        float cy = System.Math.Clamp(fy, 0.0f, ny - 1);
        float cz = System.Math.Clamp(fz, 0.0f, nz - 1);
        int ix = (int)MathF.Floor(cx);
        int iy = (int)MathF.Floor(cy);
        int iz = (int)MathF.Floor(cz);
        float tx = System.Math.Clamp(cx - ix, 0.0f, 1.0f);
        float ty = System.Math.Clamp(cy - iy, 0.0f, 1.0f);
        float tz = System.Math.Clamp(cz - iz, 0.0f, 1.0f);
        return (ix, iy, iz, tx, ty, tz);
    }

    private float PhiAt(int i, int j, int k)
    {
        int nx = (int)Config.Nx;
        int ny = (int)Config.Ny;
        int nz = (int)Config.Nz;
        if (i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz)
            return Phi[(k * ny + j) * nx + i];
        return Config.Boundary switch
        {
            FieldBoundary.Dirichlet => 0.0f,
            FieldBoundary.Neumann => Phi[(System.Math.Clamp(k, 0, nz - 1) * ny + System.Math.Clamp(j, 0, ny - 1)) * nx + System.Math.Clamp(i, 0, nx - 1)],
            _ => Phi[(Mod(k, nz) * ny + Mod(j, ny)) * nx + Mod(i, nx)],
        };
    }

    private float CellPhi(int i, int j, int k)
    {
        int nx = (int)Config.Nx;
        int ny = (int)Config.Ny;
        int nz = (int)Config.Nz;
        return Phi[(System.Math.Clamp(k, 0, nz - 1) * ny + System.Math.Clamp(j, 0, ny - 1)) * nx + System.Math.Clamp(i, 0, nx - 1)];
    }

    private Vector3 CellField(int i, int j, int k)
    {
        float ex = -(PhiAt(i + 1, j, k) - PhiAt(i - 1, j, k)) / (2.0f * Config.Dx);
        float ey = -(PhiAt(i, j + 1, k) - PhiAt(i, j - 1, k)) / (2.0f * Config.Dx);
        float ez = -(PhiAt(i, j, k + 1) - PhiAt(i, j, k - 1)) / (2.0f * Config.Dx);
        return new Vector3(ex, ey, ez);
    }

    private static long WrapOrClamp(long i, long n, FieldBoundary boundary)
    {
        if (boundary == FieldBoundary.Periodic) return Mod(i, n);
        return System.Math.Clamp(i, 0, n - 1);
    }

    private static int Mod(long a, long n) => (int)((a % n + n) % n);

    private static bool IsFinite(Vector3 v) =>
        float.IsFinite(v.X) && float.IsFinite(v.Y) && float.IsFinite(v.Z);
}



