# Audit register (2026-10-23)

A full-codebase audit (math, collision, dynamics, joints, snapshot, FFI,
subsystems, quantum, release/CI, docs). This file is the working register:
every item is `confirmed` (verified by reading the code), `reproduced`
(a failing test now exists), `fixed`, `not a defect`, or
`documented limitation`.

**Rule:** no fix lands without a test that fails before it and passes
after. A reported finding that cannot be reproduced is recorded as
`not a defect` and left alone.

**Hash policy:** the pinned trajectory hashes may move only for a
genuine correctness fix, and every move is recorded in
`PERF_BASELINE.md` with the old and new value, the reason, and the
regression that proves the new value is more correct. Windows and Linux
must agree before the new value is recorded.

## Phase 0 — guardrails

| Item | Status |
|---|---|
| This register | done |
| Version to `0.5.0-dev` (post-0.4.0 features, semver minor) | done |
| ABI layout asserts for every FFI struct + C# marshal check | Phase 3 |
| `joint.id == index` invariant asserts | Phase 2 |

## Phase 1 — P0 simulation correctness

| Item | Location | Status |
|---|---|---|
| Complex `sin`/`cos` use `conj` where they need `e^{-iz}` | `math/complex.rs:226-240`, C# mirror | fixed (M-block tests) |
| Dynamic `ConvexHull` has stub 1 kg / identity mass | `dynamics/body.rs:826` | fixed by rejecting dynamic hulls; use `ConvexMesh` (decision b) |
| 3D contact incident quad is a bowtie | `collision/contact3d.rs:136`, C# mirror | confirmed |
| CCD swept gate skips every static pair | `dynamics/world.rs:766,2701` | confirmed |
| 2D tier path `unwrap()`s a shapeless body | `dynamics/world.rs:1934` | fixed |

## Phase 2 — joints and solver (P1, confirmed)

- 3D `relative_quat` is world-relative, used as A-local (`joint3.rs:1089`).
- Motors: warm start applies a stale impulse after the budget drops to 0;
  limits likewise (`joint.rs:458`, `:553`; 3D mirrors).
- `remove_body` never renumbers joint ids / `next_joint_id`.
- Limit impulses are bilateral; joint Baumgarte bias is uncapped.
- Restitution target computed after warm start; early-exit tolerance absolute.
- Events cleared per CCD substep; sleepers not woken by cached-contact partners.

## Phases 3–7

Rolling; see `CHANGELOG.md` for what has landed. Reported-but-unverified
items stay in the audit notes until a failing test proves them.

## Known limitations (not defects)

- Convex hull/mesh are not solid for point/closest queries; hull/mesh
  miss rays (documented).
- 3D character body is a sphere; capsule needs a capsule distance query.
- No step-up; no joint compliance/bending; no TOI CCD.
- Queries are O(body count) with an AABB prefilter but no index.
