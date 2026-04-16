## Sprint 13 — Completed
- [x] IO-09: DICOM read metadata slice (series-level capture, per-slice geometry, read-only API)
- [x] IO-05: MINC2 format reader (consus-hdf5 HDF5 parsing, dimension metadata, spatial derivation, datatype conversion)
- [x] IO-05: MINC2 format writer (low-level HDF5 binary construction, contiguous f32 LE storage)
- [x] consus integration: added consus-hdf5, consus-core, consus-io, consus-compression workspace dependencies
- [x] Module wiring: minc format module registered in format/mod.rs and re-exported in lib.rs
- [x] Workspace compilation: zero errors, zero warnings
- [x] Unit tests: 27 MINC-specific tests passing (reader: data type conversion, spatial metadata, dimorder, attribute extraction; writer: validation)

---

## Sprint 11 — Completed
- [x] FLT-CAD: Curvature Anisotropic Diffusion (Alvarez et al. 1992, mean curvature motion of level sets, explicit Euler)
- [x] FLT-SATO: Sato Line Filter (Sato 1998, multi-scale Hessian curvilinear structure detection)
- [x] Hessian computation module (3-D physical-space Hessian, Cardano eigenvalue solver)
- [x] Module wiring: diffusion/curvature, vesselness/sato+hessian re-exported through filter root
- [x] Workspace compilation: zero errors
- [x] Unit tests: ritk-core 519 (unit) + 11 (integration) = 530 passing

---

## Sprint 10 — Completed
- [x] SEG-CC: Confidence Connected Region Growing (Yanowitz/Bruckstein adaptive statistics)
- [x] SEG-NC: Neighborhood Connected Region Growing (neighborhood admissibility predicate)
- [x] SEG-SK: Skeletonization (topology-preserving thinning; Zhang–Suen 2-D, directional sequential 3-D)

---

## Sprint 4 — Completed
- [x] FLT-04: RecursiveGaussianFilter (Deriche IIR, derivative orders 0/1/2)
- [x] FLT-08: CannyEdgeDetector (Gaussian + gradient + NMS + hysteresis)
- [x] FLT-10: LaplacianOfGaussianFilter (separable Gaussian + Laplacian)
- [x] FLT-09: Grayscale morphological filters (erosion, dilation)
- [x] SEG-THR: Additional thresholds (Li, Yen, Kapur, Triangle)
- [x] SEG-08: K-Means clustering segmentation (k-means++ init, Lloyd's iteration)
- [x] SEG-07: Watershed segmentation (Meyer flooding)
- [x] STA-04: Nyúl-Udupa histogram standardization (train/apply workflow)
- [x] STA-06: MAD noise estimation (unmasked + masked)
- [x] STA-08: PSNR and SSIM comparison metrics
- [x] GAP-R07: BSpline FFD registration pipeline (multi-resolution, bending energy)
- [x] Module root updates (filter, edge, segmentation, statistics, registration)
- [x] Workspace compilation: zero errors, zero warnings
- [x] Unit tests: 390 (ritk-core) + 121 (ritk-registration) = 511+ passing

## Sprint 5 — Completed
- [x] SEG-06: Level set segmentation (GAC, Chan-Vese)
- [x] FLT-09: Sobel gradient filter
- [x] PY-06: Extended Python segmentation API (16 functions)
- [x] PY-05 partial: Extended Python filter API (6 new functions, 14 total)
- [x] FLT-03/FLT-04: Native median/bilateral confirmed (already existed in core)
- Verification: workspace compiles (zero errors/warnings), 671+ tests passing

## Sprint 6 — Completed
- [x] GAP-R01: Multi-Resolution SyN (coarse-to-fine pyramid, inverse consistency enforcement)
- [x] GAP-R01b: BSplineSyN (B-spline parameterized velocity fields, bending energy regularization)
- [x] GAP-R03: LDDMM registration (geodesic shooting via EPDiff, Gaussian RKHS kernel)
- [x] GAP-R05: Composite transform serialization (JSON, TransformDescription enum, round-trip file I/O)
- [x] IO-07: TIFF/BigTIFF reader/writer (multi-page z-stack, u8/u16/u32/f32/f64 support)
- [x] PY-05: Python registration API completion (BSpline FFD, Multi-Res SyN, BSpline SyN, LDDMM — 8 total registration functions)
- [x] Module root updates (diffeomorphic, lddmm, transform, io/format, python/registration)
- [x] Workspace compilation: zero errors, zero warnings
- [x] Unit tests: ritk-core 421, ritk-registration 150+, ritk-io 50+ passing

## Sprint 7 — Completed
- [x] GAP-R04: Groupwise/Atlas Registration (iterative template building via SyN)
- [x] GAP-R06: Joint Label Fusion (Wang 2013) + Majority Voting
- [x] IO-MGH: MGZ/MGH Format Reader/Writer (FreeSurfer, gzip support)
- [x] SEG-DT: Euclidean Distance Transform (Meijster 2000, O(N) linear time)
- [x] STA-09: White Stripe Normalization (Shinohara 2014, KDE-based peak detection)
- [x] PY-STAT: Python Statistics/Normalization/Comparison API (13 functions)
- [x] Module wiring: atlas, distance_transform, white_stripe, mgh, statistics across all crates
- [x] Workspace compilation: zero errors, zero warnings
- [x] Unit tests: ritk-core 454, ritk-registration 162, ritk-io 79 passing

## Sprint 8 — Completed
- [x] IO-06: VTK image format reader/writer (legacy structured points, ASCII/BINARY, big-endian binary handling)
- [x] IO-08: JPEG 2D support (grayscale read/write, represented as shape `[1, height, width]`, write rejects `nz != 1`)
- [x] PY-07: CLI tooling completion
- [x] CLI filter coverage: median, bilateral, canny, sobel, log, recursive-gaussian
- [x] CLI segmentation coverage: li, yen, kapur, triangle, watershed, kmeans, distance-transform
- [x] CLI registration coverage: demons and syn
- [x] CLI stats subcommand: summary, dice, hausdorff, psnr, ssim
- [x] PY-08: Type stubs / py.typed packaged under `crates/ritk-python/python/ritk/`
- [x] Python registration additions: atlas building, majority-vote fusion, joint label fusion
- [x] Python IO additions: composite transform JSON read/write
- [x] Workspace verification: `cargo test --workspace` reached passing `ritk-cli` suite (107 tests) before timeout; prior attached report records full workspace pass (864 tests, 0 failed)
- [x] Bounded verification note: current local reruns hit build/cache lock and timeout during full-workspace execution; no failing diagnostics were observed in captured output

## Remaining Backlog
- [x] IO-05: MINC format reader/writer (Sprint 12 — consus pure-Rust HDF5 integration)
- [ ] GAP-R02b: Diffeomorphic Demons exact inverse
- [x] FLT: Curvature anisotropic diffusion (Alvarez et al. 1992 mean curvature motion)
- [x] FLT: Sato line filter (Sato 1998 multi-scale Hessian line detection)
- [x] IO-07b: Analyze format reader/writer
- [x] SEG: Confidence connected region growing (Sprint 10)
- [x] SEG: Neighborhood connected region growing (Sprint 10)
- [x] SEG: Skeletonization (Sprint 10)
- [ ] CI: nextest, clippy, fmt enforcement
- [ ] CI: dependency version alignment checks
- [ ] Python: wrap long-running PyO3 calls with `py.allow_threads`
- [ ] CI: add maturin build + `import ritk` smoke test

## Verification Policy
- All tests must assert computed VALUES, not just Result/Option variants
- cargo check --workspace --tests: zero errors required before commit
- Targeted test runs for impacted crates before push
- Use bounded verification first; if a full-workspace run times out, record the completed subset and the blocking condition