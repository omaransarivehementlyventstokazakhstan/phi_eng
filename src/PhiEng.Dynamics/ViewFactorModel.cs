namespace PhiEng.Dynamics;

/// <summary>
/// Which exchange model the body-to-body radiation uses. Mirrors the Rust
/// <c>ViewFactorModel</c> in crates/dynamics/src/thermal.rs.
/// </summary>
public enum ViewFactorModel
{
    /// <summary>
    /// Historical model: the view factor is <c>min((r_j / d)^2, 1)</c> and
    /// depends on which body of the pair is visited first.
    /// </summary>
    Legacy = 0,

    /// <summary>
    /// Order-independent model: the view factor is
    /// <c>min(r_i * r_j / d^2, 1)</c> and the exchange rate uses the mean
    /// of the two emitting areas. For two equal-sized bodies it agrees
    /// with <see cref="Legacy"/> to within a rounding step.
    /// </summary>
    Symmetric = 1,
}
