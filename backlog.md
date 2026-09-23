# RITK execution backlog

<a id="RITK-CI-PYTHON-GATE-001"></a>
## RITK-CI-PYTHON-GATE-001 â€” Preserve required checks under path selection [patch]
- Status: in-progress; priority: P0; owner: RITK CI; integrator: root; branch: `docs/ritk-done-queue-reconcile-001`; delivery: PR [#618](https://github.com/ryancinsight/ritk/pull/618), merge `641a97ad5f5f0b86fdbccf0e840d64d71d37d0a8`; last-update: 2026-09-23.
- Outcome: required contexts report on every pull request while path selection skips unrelated build and test work.
- Scope: `.github/workflows/ci.yml`, `.github/workflows/python_ci.yml`, and this backlog item; preserve the current ruleset contexts.
- Acceptance: docs-only pull requests report the expanded Python 3.12/Ubuntu matrix context; relevant Rust and Python inputs execute their suites; unrelated heavy steps skip; hosted checks pass without changing required context names.

<a id="RITK-BROWSER-READ-001"></a>
## RITK-BROWSER-READ-001 â€” Reproduce and close WebKit study reads
- Status: blocked; compacted 2026-09-18; full delivery history remains in git.
- Scope: saved public MRI-DIR chooser replay and cross-engine gallery; provider/host fixes stay with their owners.
- Acceptance: read-path cause and production fix, 94 file hashes, three exact pixel oracles, bounded rejections and clean sessions on Chromium, Firefox and WebKit.
- Risk: [patch]; dependency: [Metis read diagnosis](../metis/backlog.md#METIS-BROWSER-READ-001).
- Delivery: RITK PR [#422](https://github.com/ryancinsight/ritk/pull/422), merge `18ee4e55b`; the current failure evidence is recorded and the external WebKit authorization residual remains blocked.
- Evidence: current hosted [run 35759891764](https://github.com/ryancinsight/ritk/actions/runs/35759891764) builds RITK `db17163baf2947e6e8d163e2750c4fec7f93b671` against Metis `1b10541c2ef7a849e6ff66a3c778874bdf96de7b`; Chromium and Firefox pass the 94-file real MRI study, three exact pixel oracles and lifecycle checks; WebKit accepts the chooser but denies the first bounded read under Safari 26.6.2, with diagnostics artifact [10709802466](https://github.com/ryancinsight/ritk/actions/runs/35759891764/artifacts/10709802466) preserved in the provenance JSON.
- Diagnosis: the current WebKit trace shows the sandbox rejecting the bounded whole-file read after chooser acceptance; earlier isolated one-file/full-batch probes also failed across the available browser read paths. DICOM stays in RITK.
- Verification: locked `ritk-snap` nextest 487/487, strict native/WASM Clippy and checks, formatting, rustdoc and lockfile validation pass; [proof and log hashes](docs/manual/images/dicom-metis-real-browser-mri-cross-engine.json).
- Blocker: exact SafariDriver/WebKit selected-file authorization defect remains external; current run 35759891764 reproduces the denial after SafariDriver accepts all 94 files (artifact [10709642592](https://github.com/ryancinsight/ritk/actions/runs/35759891764/artifacts/10709642592), diagnostics [10709802466](https://github.com/ryancinsight/ritk/actions/runs/35759891764/artifacts/10709802466)). Re-open when the corrected browser/runner path grants real-file reads; application byte-read APIs cannot grant that access.

<a id="RITK-BROWSER-WEBGPU-001"></a>
## RITK-BROWSER-WEBGPU-001 â€” Demonstrate browser WebGPU presentation [arch] [minor]
- Status: blocked; compacted 2026-09-18; full delivery history remains in git.
- Scope: replay the saved public MRI-DIR study through the RITK-owned `?renderer=webgpu` page and retain actual canvas/window evidence; RITK owns DICOM decoding and clinical pixels, while MÃ©tis remains the format-neutral canvas host.
- Acceptance: a configured browser runner reports an adapter, presents the three saved-study canvases, records revision-bound PNGs and semantic attributes, and completes bounded teardown without a raster fallback.
- Blocker: hosted Chromium in run [35759891764](https://github.com/ryancinsight/ritk/actions/runs/35759891764) reports no WebGPU adapter; the setup error and failure capture artifact [10710217395](https://github.com/ryancinsight/ritk/actions/runs/35759891764/artifacts/10710217395) are preserved in [`dicom-metis-real-browser-mri-webgpu-failure.png`](docs/manual/images/dicom-metis-real-browser-mri-webgpu-failure.png).

<a id="MIG-439-03"></a>

## MIG-439-03 â€” Replace remaining Burn NdArray backend aliases with Atlas-backed surfaces

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Atlas-backed surfaces. RESCOPED (was READY; original acceptance criteria do not hold â€” see Sprint 465 finding).** **Correction (Sprint 465, evidence-based)**: investigated the strongest candidate crate (`ritk-jpeg`, smallest burn_ndarray footprint, already has a parallel Coeus reader) to execute thâ€¦

<a id="PERF-435-01"></a>

## PERF-435-01 â€” Route MSE through fused interpolation. PARTIAL.

- Status: in-progress; retained from the historical execution section; full detail remains in git.

- Scope: Generalized `ritk_interpolation::transform_and_interpolate` over spatial dimensionality, generalized the OOB mask helper over the image shape length, and routed `MeanSquaredError` through the fused transform-to-index-to-linear interpolation path. Evidence tier: value-semantic nextest and focused tiâ€¦

<a id="PROVIDER-420-01"></a>

## PROVIDER-420-01 â€” Hermes complex dispatch bound cleanup. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: The local Atlas provider graph exposed that Hermes complex SIMD operations still require `Neg` at the complex-operation dispatch surface after broader unsigned scalar support. A minimal local fix passes `cargo check -p hermes-simd --all-targets`; full provider rustfmt is still blocked by unrelatedâ€¦

<a id="PERF-419-01"></a>

## PERF-419-01 â€” Registration test runtime budget breach. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 419's `ritk-registration` nextest gate passed but exposed integration tests above the 30s slow budget, including 100s, 146s, and 193s rows. Treat this as a real performance defect to profile; do not weaken or skip those tests.

<a id="COEUS-406-01"></a>

## COEUS-406-01 â€” Fix dirty Coeus autograd provider compile break. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: RITK doctest/doc gates against the current local Atlas stack are blocked after refreshing Coeus path packages to `0.2.6`: `D:\atlas\repos\coeus` is dirty on `test/cuda-parity-suite`, and `coeus-autograd` fails to compile in shape/reduction ops. This must be fixed in Coeus before RITK can claim docsâ€¦

<a id="PERF-406-02"></a>

## PERF-406-02 â€” Registration test runtime budget breach. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 406's touched-package `nextest` gate passed but exposed registration tests above the 30s slow budget, including 93s, 129s, and 183s rows. Treat this as a real performance defect to profile; do not weaken or skip those tests.

<a id="PERF-387-02"></a>

## PERF-387-02 â€” Continue flat-buffer memory-efficiency audit. IN PROGRESS.

- Status: in-progress; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 387 flattened `VectorConfidenceConnected` covariance/inverse matrices and removed the B-spline legacy placeholder. Sprint 389 flattened `InverseDisplacementField` TPS spline/affine coefficient blocks after the solve. Sprint 390 flattened TIFF grayscale/RGB page accumulation by removing `Vec<â€¦

<a id="MIG-387-01"></a>

## MIG-387-01 â€” Atlas crate migration audit.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Continue replacing production `nalgebra`/`ndarray`/`burn` surfaces with `leto`/`coeus`/ `hephaestus` only after each target operation has a verified equivalent contract and focused differential tests. Do not remove boundary dependencies used only for file-format interop or external framework contraâ€¦

<a id="MIG-387-02"></a>

## MIG-387-02 â€” Spatial Leto SSOT migration. IN PROGRESS.

- Status: in-progress; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 408 migrates `ritk-spatial` storage to Leto fixed vectors/matrices and removes direct `nalgebra` dependencies from `ritk-core`, `ritk-metaimage`, `ritk-nrrd`, `ritk-nifti`, and `ritk-mgh` spatial direction setup. Sprint 409 moves DICOM IO, MINC, and filter spatial-transform consumers onto `Dâ€¦

## Sprint 342 (Phase 20) â€” Coeus Migration Readiness Audit

- Status: in-progress; historical sprint retained only for unresolved rows.

- **Status**: In Progress

- | MIG-342-04 | RITK-owned tensor contract over Coeus CPU backend | **Open** |

- | GPU-342-05 | Coeus WGPU differential test harness for RITK operation subset | **Open** |

- | REG-342-06 | Registration autodiff tape continuity proof/test under Coeus | **Open** |

- | MODEL-342-07 | `ritk-model` Coeus module/parameter/3-D convolution migration design | **Open** |

- | PY-342-08 | Python binding conversion plan over Coeus-backed Rust core | **Open** |

- The next implementation stage is not a dependency swap. It is the RITK tensor

- ### Residual risks



## Sprint 332 (0.50.95) â€” Documentation Compaction + Structural Audit + Benchmark

- Status: in-progress; historical sprint retained only for unresolved rows.

- **Status**: In Progress

- | BENCH-332-03 | `STACK_WEIGHTS_CAPACITY=32` Criterion benchmark â€” measure AVX2 speedup vs 8-entry version | **Open** |

- | GPU-332-04 | Evaluate `sparse.rs` GPU-backend potential (Burn autodiff scatter compatibility, custom kernel feasibility) | **Open** |

- | CRLF-332-05 | Git CRLF normalization (`git add --renormalize`) â€” blocked by missing test data files | **Blocked** |



<a id="RITK-GAP-2026-08-20-01"></a>
## RITK-GAP-2026-08-20-01 [major][arch] â€” collapse the dual `X` / `X_native` surface
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-02"></a>
## RITK-GAP-2026-08-20-02 [minor] â€” fuzz the sixteen format parsers
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-03"></a>
## RITK-GAP-2026-08-20-03 [patch] â€” retire the GPU naming and the unbacked accelerator claims
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-04"></a>
## RITK-GAP-2026-08-20-04 [patch] â€” evict the 1.6 GB tracked binary payload
- Status: in-progress; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-05"></a>
## RITK-GAP-2026-08-20-05 [patch] â€” derive the escalated test budgets and sweep the dead filters
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-06"></a>
## RITK-GAP-2026-08-20-06 [patch] â€” raise the lint and documentation floor
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-07"></a>
## RITK-GAP-2026-08-20-07 [patch] â€” restore the CHANGELOG version axis
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-08"></a>
## RITK-GAP-2026-08-20-08 [patch] â€” write the registration and dispatch book chapters
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-09"></a>
## RITK-GAP-2026-08-20-09 [patch] â€” derive or remove the MI subsample stride
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-DOCS-QUEUE-001"></a>
## RITK-DOCS-QUEUE-001 â€” Remove merged work from the active queue [patch]
- Status: review; priority: P1; owner: RITK documentation; integrator: root; branch: `docs/ritk-done-queue-reconcile-001`; last-update: 2026-09-23.
- Outcome: the RITK queue lists unresolved work only.
- Scope: `backlog.md` and ADR delivery references to completed queue entries.
- Acceptance: remove the 34 previously merged entries and newly completed RITK-ALLOW-001; retain all 20 original unresolved records and RITK-CI-PYTHON-GATE-001 until its docs-only regression check passes; ADR references resolve and board checks pass.

<a id="RITK-MANUAL-PROVENANCE-001"></a>
## RITK-MANUAL-PROVENANCE-001 â€” Bind MRI resource metadata to its PNG [patch]
- Status: todo; priority: P1; owner: RITK documentation; integrator: unclaimed; last-update: 2026-09-23; dependency delivery: [PR #605](https://github.com/ryancinsight/ritk/pull/605).
- Outcome: the public MRI manual PNG and its resource manifest report matching output digest and size while retaining distinct source-capture evidence.
- Scope: docs/manual/images/dicom-metis-real-mri-resource.json, a focused integrity test in scripts/tests/, and affected manual provenance text.
- Acceptance: output SHA-256 and byte count match the tracked PNG, capture SHA-256 and byte count remain unchanged, and provenance tests pass.
