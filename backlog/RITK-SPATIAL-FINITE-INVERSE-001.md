<a id="RITK-SPATIAL-FINITE-INVERSE-001"></a>

## RITK-SPATIAL-FINITE-INVERSE-001 — Reject non-finite direction inversion [patch] — todo

- outcome: `Direction::try_inverse` returns `None` when its input or normalization/elimination is non-finite; Cartesian geometry reports the existing typed error instead of retaining an invalid transform pair.
- basis: current main `2775dacf98854702969ee110346398057e6b79a5`, Rust 1.97.0, aarch64-apple-darwin. An isolated checkout builds against main's unchanged canonical external dependency records, without a provider migration or registry publication.
- reproduced: three current-source regression tests fail before the repair. A one-dimensional NaN matrix returns `Some([NaN])`; a NaN shear is accepted by `CartesianGridGeometry`; finite `[[1,MAX],[-1,MAX]]` elimination returns the incorrect finite inverse `[[1,0],[0,0]]` after `MAX+MAX` overflows.
- correction: reject non-finite entries before elimination and check every normalized/eliminated matrix and inverse entry. Keep the existing pivot threshold and algorithm. Numerical failure returns `None`; no identity, zero matrix, transpose or other substitute is supplied. The existing `NonCartesianGrid::SingularDirection` variant reports that finite inversion is unavailable.
- tests: every position of ranks 1, 2 and 3 is tested with NaN and both infinities (42 cases); normalization and elimination overflow are rejected; the Cartesian error is inspected. Exact pivoted/sheared inverses and `2^1023` diagonal inversion to `2^-1023` preserve accepted finite/subnormal behavior.
- validation: 160 spatial/annotation Nextest tests pass with zero skips in 3.94 seconds under a 60-second command bound. Strict spatial all-target Clippy passes; final exact-head checks and independent review remain required. Main's existing TrackVis error attribute needs a rustfmt-only line wrap; its tokens and string are unchanged.
- version: ritk-spatial 0.4.0 to 0.4.1. Existing `^0.4.0` incoming requirements admit the patch. The lock changes only this package version and two already-locked tracing test-dependency edges; all external Git/registry records remain identical to main.
- ownership: active audit PR835 touches coordinate-map and rotation files, not this direction/grid implementation. Preserved resampler/storage regressions and their provider-integration merge remain untouched. Metis503/504/506 and the Coeus528 owner work are outside this slice.
- acceptance: current-source red/green evidence, finite-value oracles, strict local checks, independent review and required final-revision gates before an authorized merge. The separate first-upload authentication and Eunomia version question remain unresolved; this slice does not retry that graph or configure credentials. Miri remains deferred.
- status: todo
- priority: P1
- next: complete the draft review and exact-head required gates; retain any unrelated publication or CI blocker explicitly.
