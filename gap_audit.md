# RITK Gap Audit

<a id="RITK-REGISTRY-PUBLICATION-001"></a>
## RITK-REGISTRY-PUBLICATION-001: Git validation does not establish registry publication [patch]
- risk: a corrected Git revision passes native checks while the registry resolves older incompatible dependencies or lacks required packages.
- evidence: main's release package verification fails in registry Leto 0.42.0; an isolated Coeus 0.11 probe compiles but its registry plan holds registration on unpublished Consus and missing Coeus component versions.
- next: preserve active provider owners, validate the complete required registry graph, and complete approved first publications before claiming a released registration version.
- review: [RITK-REGISTRY-PUBLICATION-001](backlog/RITK-REGISTRY-PUBLICATION-001.md) records exact failure, ownership and credential boundaries.

<a id="RITK-MIND-HALF-TIE-001"></a>
## RITK-MIND-HALF-TIE-001: Descriptor rounding migration changes similarity [patch]
- risk: exact response ties change production packed descriptors and registration objectives.
- evidence: main `8af30a9a` with locked Eunomia `ee7c6911`; three failing regression tests on Rust 1.97.0 / aarch64-apple-darwin. A reachable dyadic image changes constant-image Hamming loss 28→32 and public similarity `0.5333333`→`0.46666664`.
- correction: restore binary32 half-away rounding before exact integer conversion; all 20 MIND tests pass. ADR0024 now records the tie policy. Review and full validation are tracked in [the backlog item](backlog/RITK-MIND-HALF-TIE-001.md).
- reopen: a quantizer, descriptor, or prepared/one-shot regression fails; platform-specific exponential precision must not dictate the tie policy.

<a id="RITK-PYTHON-CI-CACHE-SAVE-001"></a>
## RITK-PYTHON-CI-CACHE-SAVE-001: Branch cache archives consume the validation budget [patch]
- risk: all required test stages pass but optional gigabyte cache creation cancels the Windows Python 3.10 job and fails its aggregate gate.
- evidence: exact-head dispatch `38656789073`, attempt 1, spends at least 65 seconds in `tar`/`zstd` before the unchanged 30-minute bound; other Windows cache post-steps take 74–186 seconds and upload approximately 1.12 GB each.
- correction: use the pinned action's documented main-only `save-if` input, matching native CI; restores, test coverage, required gates, permissions and limits remain unchanged.
- review: [RITK-PYTHON-CI-CACHE-SAVE-001](backlog/RITK-PYTHON-CI-CACHE-SAVE-001.md) records measured timing and final-revision validation criteria.

<a id="RITK-PYTHON-SCRIPT-RUNTIME-001"></a>
## RITK-PYTHON-SCRIPT-RUNTIME-001: Script tests require newer Python APIs [patch]
- risk: supported Python 3.9/3.10 cells stop before wheel/parity/Rust validation; shell PATH changes can select a different interpreter and hide version defects.
- evidence: unchanged main `8af30a9a`; real Python 3.9 reproduces nine failed lock-parser subcases, followed by three gallery `write_text(newline=...)` errors after parser support is restored.
- correction: declared conditional TOML dependency and setup, portable LF writers, and interpreter/format/byte regressions; 41 script tests pass on four real interpreter versions.
- review: [RITK-PYTHON-SCRIPT-RUNTIME-001](backlog/RITK-PYTHON-SCRIPT-RUNTIME-001.md) records the separate draft, final-revision validation criteria and the distinction between draft-policy and executed CI outcomes.

<a id="PERF-432-01"></a>
## PERF-432-01: Revalidate the legacy MSE performance claim
- risk: June profiling may no longer describe the current backend and can select the wrong optimization target.
- evidence: prior profile claims predate backend and interpolation changes; no current-main reproduction is recorded.
- reopen: before changing MSE or B-spline performance, collect a new exact-main profile and replace or remove the stale claim.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
