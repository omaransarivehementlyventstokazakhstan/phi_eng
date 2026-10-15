# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

Please report security vulnerabilities by opening a private GitHub Security Advisory:

1. Go to the **Security** tab of this repository
2. Click **Report a vulnerability**
3. Fill in the details

We will acknowledge receipt within 72 hours and provide a status update within 7 days.

## Threat Model

PhiEng is a deterministic physics library. It is **not** designed to protect against:

- Malicious input crafted to exploit memory safety (all Rust code is `safe` by default; `unsafe` blocks are documented and audited per release — see `RELEASE_CHECKLIST.md` §4)
- Denial-of-service via resource exhaustion (no allocation limits on public APIs)
- Side-channel timing attacks (determinism is a feature, not a security boundary)

**Do not** use PhiEng in security-critical contexts (access control, cryptography, safety systems) without your own threat modeling and mitigations.

## Safe Usage Guidelines

- Always validate untrusted scene data before deserializing snapshots (loaders reject unknown schema versions but cannot guarantee semantic safety)
- Set `ValidationConfig::Strict` on worlds that process untrusted input
- The FFI boundary (`phieng.dll`) validates all pointer/length pairs and caps array sizes at 1,000,000 elements; callers must still respect ownership and lifetime rules
- Parallel island solve (`Solver.Parallel`) is opt-in and debug-verified for disjointness; enable only when you control the scene topology