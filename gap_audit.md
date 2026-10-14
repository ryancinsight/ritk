# RITK Gap Audit

<a id="RITK-MIND-HALF-TIE-001"></a>
## RITK-MIND-HALF-TIE-001: Descriptor rounding migration changes similarity [patch]
- risk: exact response ties change production packed descriptors and registration objectives.
- evidence: main `8af30a9a` with locked Eunomia `ee7c6911`; three failing regression tests on Rust 1.97.0 / aarch64-apple-darwin. A reachable dyadic image changes constant-image Hamming loss 28→32 and public similarity `0.5333333`→`0.46666664`.
- correction: restore binary32 half-away rounding before exact integer conversion; all 20 MIND tests pass. ADR0024 now records the tie policy. Review and full validation are tracked in [the backlog item](backlog/RITK-MIND-HALF-TIE-001.md).
- reopen: a quantizer, descriptor, or prepared/one-shot regression fails; platform-specific exponential precision must not dictate the tie policy.

<a id="PERF-432-01"></a>
## PERF-432-01: Revalidate the legacy MSE performance claim
- risk: June profiling may no longer describe the current backend and can select the wrong optimization target.
- evidence: prior profile claims predate backend and interpolation changes; no current-main reproduction is recorded.
- reopen: before changing MSE or B-spline performance, collect a new exact-main profile and replace or remove the stale claim.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
