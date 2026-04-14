# RITK Sprint Backlog

> **Artifact type:** Sprint tracking
> **Source of truth:** `gap_audit.md` (gap IDs, priorities, severity)
> **Last updated:** 2025-07-18 (post-Sprint 6)

---

## Legend

| Column   | Meaning |
|----------|---------|
| **ID**       | Gap audit identifier (`CORE-*` = foundation, others from `gap_audit.md` §8) |
| **Feature**  | Deliverable name |
| **Sprint**   | Sprint in which the item was/will be delivered |
| **Status**   | `COMPLETED` · `PLANNED` |
| **Priority** | `Critical` · `High` · `Medium` · `Low` (from gap audit severity) |
| **Notes**    | Implementation location, key decisions, residual gaps |

---

## Sprint 1 — Foundations

Core architecture: spatial types, image primitives, transforms, interpolation, registration framework, metrics, optimizers, base I/O.

| ID | Feature | Sprint | Status | Priority | Notes |
|---|---|---|---|---|---|
| CORE-01 | Spatial Types (Point, Vector, Spacing, Direction) | 1 | COMPLETED | Critical | nalgebra zero-cost type aliases; `ritk-core/src/spatial/` |
| CORE-02 | Image Type with Physical Metadata | 1 | COMPLETED | Critical | Tensor + origin/spacing/direction; `ritk-core/src/image/` |
| CORE-03 | Transform Framework (Translation, Rigid, Affine, BSpline) | 1 | COMPLETED | Critical | Trait-driven, const-generic dimension; `ritk-core/src/transform/` |
| CORE-04 | Interpolation (Linear, Nearest Neighbor) | 1 | COMPLETED | Critical | Bilinear/trilinear + nearest; `ritk-core/src/interpolation/` |
| CORE-05 | Registration Framework | 1 | COMPLETED | Critical | Metric + Optimizer + Transform loop; `ritk-registration/src/` |
| CORE-06 | Similarity Metrics (MSE, MI, NCC, LNCC, NMI, CR) | 1 | COMPLETED | Critical | Full differentiable metric suite; `ritk-registration/src/metric/` |
| CORE-07 | Optimizers (GD, Adam, Momentum, L-BFGS) | 1 | COMPLETED | Critical | Trait-driven optimizer suite; `ritk-registration/src/optimizer/` |
| IO-NII | NIfTI I/O (Reader + Writer) | 1 | COMPLETED | Critical | Via nifti-rs; `ritk-io/src/nifti/` |
| IO-DCM | DICOM Series Reader | 1 | COMPLETED | Critical | Via dicom-rs; `ritk-io/src/dicom/` |

---

## Sprint 2 — Segmentation Core + Statistics + IO Formats

Segmentation primitives, statistical foundation, MetaImage/NRRD I/O. Completed 2025-07-15.

| ID | Feature | Sprint | Status | Priority | Notes |
|---|---|---|---|---|---|
| SEG-01 | Morphological Operations (Binary Erosion, Dilation, Opening, Closing) | 2 | COMPLETED | Critical | Structuring-element based; `ritk-core/src/segmentation/morphology/` |
| SEG-02 | Connected Component Labeling | 2 | COMPLETED | Critical | Hoshen-Kopelman union-find, 6/26-connectivity; `ritk-core/src/segmentation/labeling/` |
| SEG-03 | Otsu / Multi-Otsu Thresholding | 2 | COMPLETED | Critical | Inter-class variance maximization; `ritk-core/src/segmentation/threshold/` |
| SEG-04 | Region Growing (Connected Threshold) | 2 | COMPLETED | Critical | Seed-based flood-fill; `ritk-core/src/segmentation/region_growing/` |
| STA-01 | Image Statistics API (Min, Max, Mean, Variance, Percentile) | 2 | COMPLETED | Critical | Masked statistics included; `ritk-core/src/statistics/` |
| STA-02 | Histogram Matching | 2 | COMPLETED | Critical | Quantile-quantile piecewise-linear mapping; `ritk-core/src/statistics/normalization/` |
| STA-03 | Z-score / Min-Max Normalization | 2 | COMPLETED | Critical | DL preprocessing; `ritk-core/src/statistics/normalization/` |
| STA-05 | Label Statistics | 2 | COMPLETED | High | Per-label volume, centroid, bounding box; `ritk-core/src/statistics/` |
| STA-07 | Dice / Hausdorff / Mean Surface Distance | 2 | COMPLETED | High | Segmentation evaluation metrics; `ritk-core/src/statistics/` |
| IO-01 | MetaImage (.mha/.mhd) Reader + Writer | 2 | COMPLETED | Critical | Round-trip verified, ZYX↔XYZ permutation, external data file; `ritk-io/src/format/` |
| IO-02 | NRRD Reader + Writer | 2 | COMPLETED | High | External-data-file support; `ritk-io/src/format/` |

---

## Sprint 3 — Critical Filtering + Deformable Registration + Bindings

N4, edge filters, diffusion, vesselness, Demons family, greedy SyN, Python bindings, CLI. Completed 2025-07-16.

| ID | Feature | Sprint | Status | Priority | Notes |
|---|---|---|---|---|---|
| FLT-01 | N4 Bias Field Correction | 3 | COMPLETED | Critical | Tustison 2010; B-spline Tikhonov, Wiener histogram sharpening; `ritk-core/src/filter/bias/` |
| FLT-02 | Gradient Magnitude / Sobel | 3 | COMPLETED | Critical | Central-difference with physical spacing; `ritk-core/src/filter/edge/` |
| FLT-03 | Median Filter | 3 | COMPLETED | High | Sliding-window CPU; exposed via Python binding `ritk.filter.median_filter` |
| FLT-05 | Bilateral Filter (Python binding) | 3 | COMPLETED | High | Tomasi-Manduchi 1998; Python-side `ritk.filter.bilateral_filter`. Native Rust `Image<B,D>` API remains a gap |
| FLT-06 | Frangi Vesselness Filter | 3 | COMPLETED | High | Multiscale Hessian eigenanalysis (Kopp 2008), bright/dark polarity; `ritk-core/src/filter/vesselness/` |
| FLT-07 | Anisotropic Diffusion (Perona-Malik) | 3 | COMPLETED | High | Explicit Euler, exp/quad conductance, Neumann BC; `ritk-core/src/filter/diffusion/` |
| FLT-11 | Laplacian Filter | 3 | COMPLETED | Medium | Second-order FD, one-sided at boundaries; `ritk-core/src/filter/edge/` |
| GAP-R02 | Demons Registration Family (Thirion + Diffeomorphic + Symmetric) | 3 | COMPLETED | Critical | Thirion 1998, Vercauteren 2009, Pennec 1999; scaling-and-squaring, BCH update; `ritk-registration/src/demons/` |
| GAP-R01p | Greedy SyN Registration | 3 | COMPLETED | Critical | Avants 2008; local CC, symmetric velocity fields, Gaussian regularisation; `ritk-registration/src/diffeomorphic/` |
| PY-01 | Python Module Scaffold (PyO3 + maturin) | 3 | COMPLETED | Critical | `_ritk` extension, abi3-py39 stable ABI (Python 3.9–3.14); `ritk-python/` |
| PY-02 | NumPy ↔ Image Bridge | 3 | COMPLETED | Critical | Bidirectional ndarray conversion; `ritk-python/src/image/` |
| PY-03 | Python Image I/O | 3 | COMPLETED | Critical | MetaImage/NRRD wiring; `ritk-python/src/io/` |
| PY-04 | Python Filter API | 3 | COMPLETED | High | N4, Gaussian, diffusion, gradient magnitude, vesselness; `ritk-python/src/filter/` |
| PY-07p | CLI Tooling (Initial) | 3 | COMPLETED | Medium | `ritk` binary: convert, filter, register, segment subcommands; 59 tests; `ritk-cli/` |

---

## Sprint 4 — Advanced Filters + Segmentation Expansion + BSpline FFD

Recursive Gaussian, Canny, LoG, grayscale morphology, extended thresholds, clustering, watershed, Nyúl-Udupa, noise estimation, PSNR/SSIM, BSpline FFD registration. Completed 2025-07-17.

| ID | Feature | Sprint | Status | Priority | Notes |
|---|---|---|---|---|---|
| FLT-04 | Recursive Gaussian (Deriche IIR) | 4 | COMPLETED | High | O(N) large-σ smoothing, first/second derivative orders; `ritk-core/src/filter/gaussian/` |
| FLT-08 | Canny Edge Detection | 4 | COMPLETED | Medium | Gaussian smooth → Sobel gradient → NMS → double hysteresis; `ritk-core/src/filter/edge/` |
| FLT-10 | Laplacian of Gaussian (LoG) | 4 | COMPLETED | Medium | Blob detection / edge enhancement; `ritk-core/src/filter/edge/` |
| FLT-09 | Grayscale Morphological Filters | 4 | COMPLETED | High | Grayscale erosion/dilation/opening/closing, flat SE; `ritk-core/src/filter/morphology/` |
| SEG-03b | Additional Thresholds (Li, Yen, Kapur, Triangle) | 4 | COMPLETED | Critical | Li 1998 min cross-entropy, Yen 1995 max correlation, Kapur 1985 max entropy, Zack 1977 triangle; `ritk-core/src/segmentation/threshold/` |
| SEG-08 | K-Means Clustering Segmentation | 4 | COMPLETED | Medium | Lloyd's algorithm, k-means++ init; tissue classification; `ritk-core/src/segmentation/clustering/` |
| SEG-07 | Watershed Segmentation | 4 | COMPLETED | Medium | Meyer 1994 flooding on gradient magnitude; `ritk-core/src/segmentation/watershed/` |
| STA-04 | Nyúl-Udupa Histogram Normalization | 4 | COMPLETED | High | Nyúl 1999/2000; piecewise-linear landmark percentile mapping; `ritk-core/src/statistics/normalization/` |
| STA-06 | Noise Estimation (MAD) | 4 | COMPLETED | Medium | σ̂ = 1.4826 · MAD(I); adaptive regularization parameter setting; `ritk-core/src/statistics/` |
| STA-08 | PSNR / SSIM | 4 | COMPLETED | Medium | 10 log₁₀(MAX²/MSE); Wang 2004 structural similarity; `ritk-core/src/statistics/` |
| GAP-R07 | BSpline FFD Deformable Registration | 4 | COMPLETED | High | Free-form deformation pipeline, multi-resolution control-point grid; `ritk-registration/` |

---

## Sprint 5 — Level Sets + Sobel + Extended Python APIs — Completed 2025-07-18

| ID | Feature | Sprint | Status | Priority | Notes |
|---|---|---|---|---|---|
| SEG-06 | Level Set Segmentation (Chan-Vese, Geodesic Active Contour) | 5 | COMPLETED | High | Chan-Vese 2001 region-based + GAC edge-based; sparse-field solver; `ritk-core/src/segmentation/level_set/` |
| FLT-09b | Sobel Gradient Filter | 5 | COMPLETED | High | Dedicated Sobel convolution kernel; `ritk-core/src/filter/edge/` |
| PY-06 | Extended Python Segmentation API (16 functions) | 5 | COMPLETED | High | Threshold (Otsu/Li/Yen/Kapur/Triangle/Multi-Otsu), region growing, morphology, watershed, k-means, level sets; `ritk-python/src/segmentation/` |
| PY-05p | Extended Python Filter API (6 new functions, 14 total) | 5 | COMPLETED | High | Sobel, Laplacian, LoG, Canny, grayscale morphology, recursive Gaussian added; `ritk-python/src/filter/` |
| FLT-03/04 | Native Median & Bilateral Confirmed | 5 | COMPLETED | Medium | Verified native Rust `Image<B,D>` implementations already present in `ritk-core/src/filter/` |

---

## Sprint 6 — Multi-Res SyN + BSplineSyN + LDDMM + Composite Transforms + TIFF IO + Python Registration API — Completed 2025-07-18

| ID | Feature | Sprint | Status | Priority | Notes |
|---|---|---|---|---|---|
| GAP-R01 | Multi-Resolution SyN | 6 | COMPLETED | High | Coarse-to-fine pyramid, inverse consistency; `ritk-registration/src/diffeomorphic/multires_syn.rs` |
| GAP-R01b | BSplineSyN | 6 | COMPLETED | High | B-spline velocity fields, bending energy regularization; `ritk-registration/src/diffeomorphic/bspline_syn.rs` |
| GAP-R03 | LDDMM Registration | 6 | COMPLETED | High | Geodesic shooting via EPDiff, Gaussian RKHS kernel; `ritk-registration/src/lddmm/mod.rs` |
| GAP-R05 | Composite Transform Serialization (JSON) | 6 | COMPLETED | High | TransformDescription enum, CompositeTransform, file I/O; `ritk-core/src/transform/composite_io.rs` |
| IO-07 | TIFF/BigTIFF Reader/Writer | 6 | COMPLETED | High | Multi-page z-stack, multiple pixel types; `ritk-io/src/format/tiff/` |
| PY-05 | Python Registration API (complete) | 6 | COMPLETED | High | BSpline FFD, Multi-Res SyN, BSpline SyN, LDDMM added — 8 total registration functions; `ritk-python/src/registration.rs` |

---

## Sprint 7 — Atlas Registration + Label Fusion + MINC IO (PLANNED)

| ID | Feature | Sprint | Status | Priority | Notes |
|---|---|---|---|---|---|
| GAP-R04 | Groupwise / Atlas Registration | 7 | PLANNED | High | Template building; depends on full SyN (GAP-R01) |
| GAP-R06 | Joint Label Fusion | 7 | PLANNED | Medium | Multi-atlas segmentation; depends on atlas registration (GAP-R04) |
| IO-05 | MINC Format Reader/Writer (.mnc/.mnc2) | 7 | PLANNED | High | ANTs/MNI atlas format interoperability; deferred from Sprint 6 |

---

## Sprint 8 — IO Parity + CLI/Python Completion (PLANNED)

| ID | Feature | Sprint | Status | Priority | Notes |
|---|---|---|---|---|---|
| IO-04 | MINC Format (.mnc / .mnc2) | 8 | PLANNED | High | ANTs/MNI atlas format interoperability |
| IO-05 | MGZ / MGH Format | 8 | PLANNED | Medium | FreeSurfer interoperability |
| PY-07 | CLI Tooling (Complete) | 8 | PLANNED | Medium | Extend PY-07p with full filter/registration/segmentation coverage |
| PY-08 | Type Stubs / `py.typed` | 8 | PLANNED | Medium | IDE autocomplete, mypy compatibility |

---

## Sprint 9+ — Remaining Parity (PLANNED)

| ID | Feature | Sprint | Status | Priority | Notes |
|---|---|---|---|---|---|
| IO-06 | VTK Image Format (.vtk / .vti) | 9 | PLANNED | Medium | ParaView visualization export |
| IO-07b | Analyze Format (.hdr / .img) | 9 | PLANNED | Low | Legacy format backward compatibility |
| IO-08 | JPEG 2D Support | 9 | PLANNED | Low | DICOM secondary capture compatibility |
| STA-09 | White Stripe Normalization | 9 | PLANNED | Medium | Sullivan 2017; brain-specific WM peak normalization |
| FLT-12 | Curvature Anisotropic Diffusion | 9 | PLANNED | Medium | Alvarez 1992; extends Perona-Malik (FLT-07) |
| FLT-13 | Sato Line / Hessian Blob Detection | 9 | PLANNED | Medium | Extends Frangi vesselness (FLT-06) |
| GAP-R02b | Diffeomorphic Demons exact inverse | 9 | PLANNED | Medium | ICC via exact field inversion (iterative Newton) |

---

## Summary

| Sprint | Theme | Items | Status |
|---|---|---|---|
| 1 | Foundations (core types, transforms, interpolation, registration, I/O) | 9 | COMPLETED |
| 2 | Segmentation Core + Statistics + IO Formats | 11 | COMPLETED |
| 3 | Critical Filtering + Deformable Registration + Python/CLI Bindings | 14 | COMPLETED |
| 4 | Advanced Filters + Segmentation Expansion + BSpline FFD | 11 | COMPLETED |
| 5 | Level Sets + Sobel + Extended Python APIs | 5 | COMPLETED |
| 6 | Multi-Res SyN + BSplineSyN + LDDMM + Composite Transforms + TIFF IO + Python Registration API | 6 | COMPLETED |
| 7 | Atlas Registration + Label Fusion + MINC IO | 3 | PLANNED |
| 8 | IO Parity + CLI/Python Completion | 4 | PLANNED |
| 9+ | Remaining Parity (legacy IO, additional filters, Demons exact inverse) | 7 | PLANNED |
| **Total** | | **70** | **56 done · 14 planned** |

---

## Residual Gaps (not yet scheduled)

These items are tracked in `gap_audit.md` but do not yet have sprint assignments:

- **Confidence Connected / Neighborhood Connected Region Growing** — extensions to SEG-04
- **Sparse Field Level Set Solver** — performance-critical narrow-band evolution (Whitaker 1998)
- **Vector Anisotropic Diffusion** — tensor-field variant of Perona-Malik
- **Huang Fuzzy Thresholding** — fuzzy membership threshold (Huang & Wang 1995)
- **Label Voting / Label Dilation** — morphological label propagation
- **Skeletonization** — topology-preserving thinning
- **Hole Filling** — geodesic dilation constrained by mask