using System.Runtime.CompilerServices;

// Test-only visibility: F29 exercises raw shape-tag rejection through the
// P/Invoke surface, which needs the world handle + add_body entry points.
// Production consumers use PhiNativeWorld(3) and never see internals.
[assembly: InternalsVisibleTo("PhiEng.Tests")]
