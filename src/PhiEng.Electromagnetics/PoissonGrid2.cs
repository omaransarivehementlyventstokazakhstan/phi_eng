using System.Numerics;

namespace PhiEng.Electromagnetics;

/// <summary>
/// The 2D electrostatic grid: red-black SOR for
/// <c>nabla^2 phi = -rho / (eps0 eps_r)</c>. Mirrors Rust
/// <c>PoissonGrid2</c>. The 2D model is the line-charge law (1/r field);
/// the pairwise solver is the 3D point-charge law. See the plan's
/// limitation notes.
/// </summary>
public sealed class PoissonGrid2
{
    /// <summary>The grid configuration.</summary>
    public FieldConfig2 Config { get; }
    /// <summary>The Coulomb constant the grid was solved with.</summary>
    public float CoulombK { get; }
    /// <summary>The solved potential, row-major <c>(y, x)</c>.</summary>
    public float[] Phi { get; private set; }
    /// <summary>The deposited charge density, row-major <c>(y, x)</c>.</summary>
    public float[] Rho { get; }

    /// <summary>Build a grid, validating its configuration.</summary>
    public PoissonGrid2(FieldConfig2 config, float coulombK)
    {
        if (config.Nx == 0 || config.Ny == 0) throw new ArgumentException("empty grid", nameof(config));
        if (!float.IsFinite(config.Dx) || config.Dx <= 0.0f || !IsFinite(config.Origin))
            throw new ArgumentException("bad grid spacing", nameof(config));
        int cells = checked((int)((long)config.Nx * config.Ny));
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

    /// <summary>Deposit a point charge with bilinear weights.</summary>
    public void DepositCharge(Vector2 pos, float charge)
    {
        if (!IsFinite(pos) || !float.IsFinite(charge) || charge == 0.0f) return;
        long nx = Config.Nx;
        long ny = Config.Ny;
        float fx = (pos.X - Config.Origin.X) / Config.Dx;
        float fy = (pos.Y - Config.Origin.Y) / Config.Dx;
        if (!float.IsFinite(fx) || !float.IsFinite(fy)) return;
        float ix = MathF.Floor(fx);
        float iy = MathF.Floor(fy);
        float tx = fx - ix;
        float ty = fy - iy;
        long i0 = (long)ix;
        long j0 = (long)iy;
        float density = charge / (Config.Dx * Config.Dx);
        Accumulate(i0, j0, (1.0f - tx) * (1.0f - ty), density, nx, ny);
        Accumulate(i0 + 1, j0, tx * (1.0f - ty), density, nx, ny);
        Accumulate(i0, j0 + 1, (1.0f - tx) * ty, density, nx, ny);
        Accumulate(i0 + 1, j0 + 1, tx * ty, density, nx, ny);
    }

    private void Accumulate(long i, long j, float w, float density, long nx, long ny)
    {
        if (w == 0.0f) return;
        long ii = WrapOrClamp(i, nx, Config.Boundary);
        long jj = WrapOrClamp(j, ny, Config.Boundary);
        Rho[jj * nx + ii] += density * w;
    }

    /// <summary>Solve with red-black SOR. Deterministic iteration order.</summary>
    public void Solve()
    {
        int nx = (int)Config.Nx;
        int ny = (int)Config.Ny;
        float dx2 = Config.Dx * Config.Dx;
        float scale = dx2 / (ElectroConstants.Eps0(CoulombK) * Config.EpsR);
        for (uint it = 0; it < Config.Iterations; it++)
        {
            for (int color = 0; color < 2; color++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        if ((i + j) % 2 != color) continue;
                        int idx = j * nx + i;
                        float left = PhiAt(i - 1, j);
                        float right = PhiAt(i + 1, j);
                        float down = PhiAt(i, j - 1);
                        float up = PhiAt(i, j + 1);
                        float target = (left + right + down + up + scale * Rho[idx]) * 0.25f;
                        float old = Phi[idx];
                        Phi[idx] = old + Config.Omega * (target - old);
                    }
                }
            }
        }
    }

    /// <summary>The potential at a world point (bilinear).</summary>
    public float PotentialAt(Vector2 pos)
    {
        var (i0, j0, tx, ty) = Locate(pos);
        float p00 = CellPhi(i0, j0);
        float p10 = CellPhi(i0 + 1, j0);
        float p01 = CellPhi(i0, j0 + 1);
        float p11 = CellPhi(i0 + 1, j0 + 1);
        float a = p00 + (p10 - p00) * tx;
        float b = p01 + (p11 - p01) * tx;
        return a + (b - a) * ty;
    }

    /// <summary>The electric field at a world point: -grad(phi), bilinear.</summary>
    public Vector2 FieldAt(Vector2 pos)
    {
        var (i0, j0, tx, ty) = Locate(pos);
        Vector2 e00 = CellField(i0, j0);
        Vector2 e10 = CellField(i0 + 1, j0);
        Vector2 e01 = CellField(i0, j0 + 1);
        Vector2 e11 = CellField(i0 + 1, j0 + 1);
        Vector2 a = e00 + (e10 - e00) * tx;
        Vector2 b = e01 + (e11 - e01) * tx;
        return a + (b - a) * ty;
    }

    /// <summary>The electrostatic energy <c>0.5 sum(rho phi) dx^2</c>.</summary>
    public float Energy()
    {
        float sum = 0.0f;
        for (int i = 0; i < Rho.Length; i++) sum += Rho[i] * Phi[i];
        return 0.5f * sum * Config.Dx * Config.Dx;
    }

    private (int, int, float, float) Locate(Vector2 pos)
    {
        int nx = (int)Config.Nx;
        int ny = (int)Config.Ny;
        float fx = (pos.X - Config.Origin.X) / Config.Dx;
        float fy = (pos.Y - Config.Origin.Y) / Config.Dx;
        if (!float.IsFinite(fx) || !float.IsFinite(fy)) return (0, 0, 0.0f, 0.0f);
        float cx = System.Math.Clamp(fx, 0.0f, nx - 1);
        float cy = System.Math.Clamp(fy, 0.0f, ny - 1);
        int ix = (int)MathF.Floor(cx);
        int iy = (int)MathF.Floor(cy);
        float tx = System.Math.Clamp(cx - ix, 0.0f, 1.0f);
        float ty = System.Math.Clamp(cy - iy, 0.0f, 1.0f);
        return (ix, iy, tx, ty);
    }

    private float PhiAt(int i, int j)
    {
        int nx = (int)Config.Nx;
        int ny = (int)Config.Ny;
        if (i >= 0 && j >= 0 && i < nx && j < ny) return Phi[j * nx + i];
        return Config.Boundary switch
        {
            FieldBoundary.Dirichlet => 0.0f,
            FieldBoundary.Neumann => Phi[System.Math.Clamp(j, 0, ny - 1) * nx + System.Math.Clamp(i, 0, nx - 1)],
            _ => Phi[Mod(j, ny) * nx + Mod(i, nx)],
        };
    }

    private float CellPhi(int i, int j)
    {
        int nx = (int)Config.Nx;
        int ny = (int)Config.Ny;
        return Phi[System.Math.Clamp(j, 0, ny - 1) * nx + System.Math.Clamp(i, 0, nx - 1)];
    }

    private Vector2 CellField(int i, int j)
    {
        float ex = -(PhiAt(i + 1, j) - PhiAt(i - 1, j)) / (2.0f * Config.Dx);
        float ey = -(PhiAt(i, j + 1) - PhiAt(i, j - 1)) / (2.0f * Config.Dx);
        return new Vector2(ex, ey);
    }

    private static long WrapOrClamp(long i, long n, FieldBoundary boundary)
    {
        if (boundary == FieldBoundary.Periodic) return Mod(i, n);
        return System.Math.Clamp(i, 0, n - 1);
    }

    private static int Mod(long a, long n) => (int)((a % n + n) % n);

    private static bool IsFinite(Vector2 v) => float.IsFinite(v.X) && float.IsFinite(v.Y);
}



