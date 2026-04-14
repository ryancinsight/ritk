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

## Sprint 5 — Next Priorities
- [ ] SEG-06: Level set segmentation (Geodesic Active Contour, Chan-Vese)
- [ ] GAP-R01: SyN improvements (multi-resolution, BSplineSyN, inverse consistency)
- [ ] GAP-R05: Composite transform serialization (HDF5/JSON)
- [ ] IO-03: TIFF/BigTIFF reader/writer
- [ ] IO-04: MINC (.mnc2) reader/writer
- [ ] IO-05: MGZ/MGH reader/writer
- [ ] PY: GIL release for long-running operations
- [ ] PY: maturin develop end-to-end verification

## Sprint 6+ — Backlog
- [ ] GAP-R03: LDDMM (geodesic shooting, EPDiff)
- [ ] GAP-R04: Groupwise/Atlas registration
- [ ] GAP-R06: Joint Label Fusion
- [ ] GAP-R02b: Diffeomorphic Demons exact inverse
- [ ] IO-06: VTK reader/writer
- [ ] IO-07: Analyze reader/writer
- [ ] IO-08: JPEG reader
- [ ] STA: White stripe normalization
- [ ] FLT: Curvature anisotropic diffusion
- [ ] FLT: Sato line filter
- [ ] SEG: Confidence connected region growing
- [ ] SEG: Neighborhood connected region growing
- [ ] SEG: Distance transform (Meijster)
- [ ] SEG: Skeletonization
- [ ] CI: nextest, clippy, fmt enforcement
- [ ] CI: dependency version alignment checks

## Verification Policy
- All tests must assert computed VALUES, not just Result/Option variants
- cargo check --workspace --tests: zero errors required before commit
- Targeted test runs for impacted crates before push