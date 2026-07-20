# RITK — Rust Image Toolkit

A high-performance medical image processing and registration toolkit built in
Rust, inspired by ITK concepts and integrated with the Coeus and Leto compute
stack.

## Overview

RITK provides a comprehensive framework for medical image analysis:

- **Backend-Parametric Compute**: Coeus tensor and autograd contracts execute
  over Leto-owned storage; current RITK entry points use deterministic
  sequential or Moirai-parallel CPU backends
- **Deep Module Hierarchy**: Strict DIP/SSOT/SoC/SRP architecture across workspace crates
- **Broad Format Support**: DICOM, NIfTI, MetaImage, NRRD, PNG, TIFF/BigTIFF, MGH/MGZ, VTK, JPEG
- **Classical & Deformable Registration**: Rigid, affine, B-Spline FFD, Demons, SyN, LDDMM, Atlas/Groupwise
- **Deep-Learning Registration**: TransMorph and SSMMorph through Coeus
  autodiff
- **Image Processing Pipeline**: Filtering, segmentation, statistics, normalization
- **CT Visualization Support**: Bed separation filter for CT foreground/body masking
- **Native DICOM Viewer**: `ritk-snap` desktop viewer with DICOM folder/DICOMDIR launch, MPR viewports, overlays, measurements, and tag inspection
- **Python Bindings**: PyO3 + maturin with NumPy bridge, packaged type stubs, and `py.typed`
- **CLI**: `ritk` binary with `convert`, `filter`, `register`, `segment`, and `stats` subcommands

## Crate Structure

| Layer | Crates | Responsibility |
|---|---|---|
| Domain contracts | `ritk-spatial`, `ritk-image`, `ritk-transform`, `ritk-interpolation`, `ritk-annotation` | Physical coordinates, typed images, transforms, interpolation, and annotation state |
| Operations | `ritk-filter`, `ritk-segmentation`, `ritk-morphology`, `ritk-statistics`, `ritk-tensor-ops` | Image algorithms and shared Coeus host-buffer operations |
| Registration | `ritk-registration`, `ritk-model` | Classical, deformable, differentiable, and learned registration |
| Format owners | `ritk-dicom`, `ritk-codecs`, `ritk-nifti`, `ritk-nrrd`, `ritk-metaimage`, `ritk-mgh`, `ritk-analyze`, `ritk-png`, `ritk-jpeg`, `ritk-tiff`, `ritk-minc`, `ritk-vtk` | Validated byte-level codecs and format-specific I/O |
| Integration | `ritk-io`, `ritk-core`, `ritk-wgpu-compat` | Unified I/O dispatch, public facade contracts, and graphics interop |
| Deliverables | `ritk-cli`, `ritk-snap`, `ritk-python` | CLI, native viewer, and thin PyO3 bindings |

Dependencies point inward toward domain contracts. Format crates own byte-level
parsing, `ritk-io` owns cross-format dispatch, and applications and bindings
depend on those contracts without moving domain logic into their boundaries.

### Viewer (`ritk-snap`)

`ritk-snap [PATH]` launches the native viewer directly against a DICOM folder,
a single DICOM file, a `DICOMDIR` file, or a supported medical image file. The viewer keeps DICOM
I/O in `ritk-io` and presentation logic in `ritk-snap`, with a vertical module
split for input path normalization, hanging-protocol selection, series
discovery, metadata row construction, session snapshot persistence, rendering,
tools, and egui widgets.

Current viewer capabilities include DICOM series browsing, axial/coronal/
sagittal MPR layout, modality-aware window presets, colormaps, measurement and
ROI tools, interactive segmentation label paint/erase with brush radius,
label visibility/active-label controls with undo/redo, viewport label overlays,
load-time hanging-protocol defaults for CT/MR series, linked MPR cursor
navigation across all three planes, DICOM-style patient-orientation labels,
linked-cursor HU overlay readout, linked-cursor physical LPS readout,
active-axis cine playback with FPS control, Ctrl/Cmd+scroll viewport zoom,
Ctrl/Cmd+0 zoom-to-fit,
Arrow Up/Down and Page Up/Down active-axis slice navigation,
Home/End active-axis first/last slice navigation,
tool keyboard shortcuts (L=length, A=angle, R=rect ROI, E=ellipse ROI, H=HU point, P=pan, Z=zoom, W=window/level, B=paint),
Zoom tool continuous drag zoom,
Segmentation keyboard undo/redo shortcuts,
Pan tool drag mapping (additive viewport offset) via SSOT `ui/pan`,
ROI Ellipse true pixel-mask statistics (ellipse membership test `((r−cy)/a)²+((c−cx)/b)²≤1`) via `Annotation::compute_roi_ellipse_stats`,
W/L drag mapping (horizontal width, vertical center) via SSOT `ui/window_level`,
PNG slice export, full axial/coronal/sagittal MPR PNG export, DICOM overlays,
RT-STRUCT contour overlay loading and rendering,
and a
deterministic Tags panel covering series metadata, first-slice geometry/display
tags, private scalar tags, preserved object-model nodes, and raw preserved
element byte counts. Viewer session save/load stores presentation state as
JSON, including source path, slice indices, window/level, colormap, active
tool, layout flags, overlay flags, sidebar tab, pan, and zoom.

### Browser / WASM (egui)

`ritk-snap` now exposes a wasm entrypoint for browser hosting:

- `ritk_snap::start_web(canvas_id: String)` (wasm-only, exported via `wasm-bindgen`)

The native binary (`ritk-snap`) remains desktop-only. For browser execution,
build `crates/ritk-snap` for `wasm32-unknown-unknown`, load the generated JS/WASM
bundle in a page with a `<canvas>` element, and invoke `start_web("<canvas-id>")`.

Minimal JS bootstrap pattern:

```javascript
import init, { start_web } from "./pkg/ritk_snap.js";

await init();
await start_web("ritk-snap-canvas");
```

## Features

### Core contracts

**Spatial types** — `Point<D>`, `Vector<D>`, `Spacing<D>`, `Direction<D>` backed by nalgebra.

**Image** — `ritk_image::Image<T, B, D>` carries typed Coeus storage plus
origin, spacing, and direction metadata with index-to-physical and
physical-to-index transforms.

**Transforms**

| Transform | Description |
|---|---|
| Translation | Pure translation |
| Rigid | Rotation + translation |
| Affine | Full affine (12 DOF in 3-D) |
| Scale | Axis-aligned scaling |
| Versor | Unit-quaternion rotation (3-D) |
| BSpline | Free-form deformation on a control-point lattice |
| DisplacementField | Dense voxel-wise displacement |
| ChainedTransform | Sequential composition of transforms |
| CompositeTransform | Named composite with JSON serialization |

**Interpolation**

| Method | Notes |
|---|---|
| Linear | Supports 1-D through 4-D |
| Nearest Neighbor | Supports 1-D through 4-D |
| BSpline | Cubic B-spline kernel |
| Tensor Trilinear | Optimized separable 3-D path |

**Filters**

| Category | Algorithms |
|---|---|
| Smoothing | Gaussian, Recursive Gaussian (Deriche), Median, Bilateral |
| Diffusion | Anisotropic Diffusion (Perona–Malik) |
| Edge Detection | Gradient Magnitude, Laplacian, Sobel, Canny, Laplacian of Gaussian |
| Vesselness | Frangi (Hessian-based) |
| Morphology | Grayscale Erosion, Grayscale Dilation |
| Bias Correction | N4 Bias Field Correction (B-spline fitting) |
| Resampling | Downsample, Resample, Multi-Resolution Pyramid |
| Distance Transform | Euclidean Distance Transform (Meijster et al. 2000) |

**Segmentation**

| Category | Algorithms |
| Segmentation | Algorithms |
|---|---|
| Thresholding | Otsu, Multi-Otsu, Li, Yen, Kapur, Triangle |
| Binary Morphology | Erosion, Dilation, Opening, Closing, Skeletonization, Fill Holes, Morphological Gradient |
| Labeling | Connected Components (Hoshen–Kopelman) |
| Region Growing | Connected threshold, Confidence connected, Neighborhood connected |
| Clustering | K-Means |
| Watershed | Marker-controlled watershed |
| Level Sets | Chan–Vese, Geodesic Active Contour, Shape Detection, Threshold Level Set, Laplacian Level Set |

**Statistics & Normalization**

| Category | Functions |
|---|---|
| Descriptive | Min, Max, Mean, Variance, Percentile (masked support) |
| Comparison | Dice, Hausdorff Distance, Mean Surface Distance, PSNR, SSIM |
| Normalization | Min-Max, Z-Score, Histogram Matching, Nyúl–Udupa, White Stripe (Shinohara 2014) |
| Noise | MAD-based noise estimation |

### I/O (`ritk-io`)

| Format | Read | Write |
|---|---|---|
| DICOM (series) | ✓ | ✓ |
| NIfTI (.nii/.nii.gz) | ✓ | ✓ |
| MetaImage (.mha/.mhd) | ✓ | ✓ |
| NRRD | ✓ | ✓ |
| PNG | ✓ | ✓ |
| TIFF / BigTIFF | ✓ | ✓ |
| MGH / MGZ (FreeSurfer) | ✓ | ✓ |
| VTK legacy structured points (`.vtk`) | ✓ | ✓ |
| JPEG (`.jpg`, `.jpeg`) | ✓ | ✓* |

*JPEG write support is limited to 2-D grayscale images represented in RITK as shape `[1, height, width]`.

`ritk-dicom` owns DICOM transfer-syntax classification and native pixel-codec primitives. Native Rust decode now covers uncompressed little-endian pixels, RLE Lossless, and grayscale JPEG Baseline/Extended/Lossless fragments. `dicom-rs` remains a backend adapter for compressed transfer syntaxes not yet replaced natively, including JPEG-LS, JPEG 2000, JPEG XL, and unsupported JPEG color/high-bit-depth variants.

### Registration (`ritk-registration`)

**Metrics** — MSE, Mutual Information (Standard / Mattes / NMI), NCC, LNCC, Correlation Ratio, DL losses.

**Optimizers** — Gradient Descent, Adam, Momentum, CMA-ES.

**Regularization** — Bending Energy, Curvature, Diffusion, Elastic, Total Variation.

**Registration Algorithms**

| Algorithm | Category |
|---|---|
| Kabsch SVD | Classical rigid alignment |
| MI-based rigid/affine | Classical iterative |
| Thirion Demons | Deformable |
| Diffeomorphic Demons | Deformable |
| Symmetric Demons | Deformable |
| Greedy SyN | Diffeomorphic |
| Multi-Resolution SyN | Diffeomorphic |
| BSpline SyN | Diffeomorphic |
| BSpline FFD | Deformable |
| LDDMM | Diffeomorphic |
| Groupwise Atlas | Template building (iterative SyN) |
| Joint Label Fusion | Multi-atlas segmentation (Wang 2013) |
| Majority Voting | Multi-atlas label fusion baseline |

### Deep-Learning Models (`ritk-model`)

| Model | Description |
|---|---|
| TransMorph | Transformer-based deformable registration |
| SSMMorph | Statistical shape model registration |

### Python Bindings (`ritk-python`)

PyO3 + maturin package exposing:

- Filters including Gaussian, median, bilateral, Canny, Frangi, and N4
- Segmentation spanning Otsu methods, morphology, connected components,
  watershed, and level sets
- Classical and deformable registration, atlas construction, and label fusion
- Descriptive statistics, similarity metrics, noise estimation, and
  normalization
- Validated NumPy ↔ `Image` conversion at the PyO3 ownership boundary
- Composite transform JSON I/O (`read_transform`, `write_transform`)
- Packaged `.pyi` type stubs and `py.typed`
- Format I/O for all supported formats

### CLI (`ritk-cli`)

```
ritk convert  <input> <output>          # Format conversion
ritk filter   <input> <output> [opts]   # Apply filters
ritk register <fixed> <moving> [opts]   # Run registration
ritk segment  <input> <output> [opts]   # Run segmentation
ritk stats    --input <path> [opts]     # Summary and comparison metrics
```

Current `ritk segment --method` coverage includes:

- Thresholding: `otsu`, `multi-otsu`, `li`, `yen`, `kapur`, `triangle`
- Region / labeling: `connected-threshold`, `connected-components`
- Morphology: `fill-holes`, `morphological-gradient`, `skeletonization`
- Region growing: `confidence-connected`, `neighborhood-connected`
- Clustering / topology: `kmeans`, `watershed`, `distance-transform`
- Level sets: `chan-vese`, `geodesic-active-contour`, `shape-detection`, `threshold-level-set`, `laplacian-level-set`

Selected method-specific options:

- `connected-components`: `--connectivity`
- `chan-vese`: `--mu`, `--nu`, `--lambda1`, `--lambda2`, `--epsilon`
- `geodesic-active-contour`: `--initial-phi`, `--propagation-weight`, `--curvature-weight`, `--advection-weight`, `--edge-k`, `--sigma`, `--dt`, `--level-set-max-iterations`
- `shape-detection`: `--initial-phi`, `--propagation-weight`, `--curvature-weight`, `--advection-weight`, `--edge-k`, `--sigma`, `--dt`, `--level-set-max-iterations`, `--tolerance`
- `threshold-level-set`: `--initial-phi`, `--lower-threshold`, `--upper-threshold`, `--propagation-weight`, `--curvature-weight`, `--dt`, `--level-set-max-iterations`, `--tolerance`
- `confidence-connected`: `--seed`, `--multiplier`, `--max-iterations`
- `neighborhood-connected`: `--seed`, `--lower`, `--upper`, `--neighborhood-radius`

## Usage Example

```rust,no_run
use ritk_io::{read_image_native, write_image_native};

fn main() -> anyhow::Result<()> {
    let image = read_image_native("input.nii.gz")?;
    write_image_native("roundtrip.nrrd", &image)?;
    Ok(())
}
```

## Dependencies

| Crate | Role |
|---|---|
| `coeus` | Tensor, backend, and autodiff contracts |
| `leto` | Array storage and numerical operations |
| `nalgebra` | Linear algebra, spatial types |
| `dicom` | DICOM format support |
| `nifti` | NIfTI format support |
| `moirai` | CPU parallelism and task execution |
| `mnemosyne` | Optional workspace allocator |
| `wgpu` | Viewer rendering and graphics interop |
| `apollo-fft` | FFT planning and execution |
| `pyo3` / `numpy` | Python bindings |
| `serde` | Serialization (transform I/O) |
| `anyhow` / `thiserror` | Error handling |
| `tracing` | Structured logging |

## Building

```bash
# Build all crates (release)
cargo build --release

# Run all tests through the committed nextest profile
cargo nextest run --workspace

# Build Python extension / install into current environment
cd crates/ritk-python && maturin develop --release

# Install CLI
cargo install --path crates/ritk-cli
```

Hosted workflows check out RITK at `ritk/` and invoke the Atlas-owned
`checkout-path-dependencies` composite action at an immutable Atlas commit.
The action reads `ritk/Cargo.toml` and materializes only its external sibling
path dependencies at the exact Atlas gitlinks. Provider URLs and revisions do
not have a second RITK-owned list.

## Development

### Recent Sprints

- **Sprint 338** (v0.51.6, `ritk-core` 0.6.0): `value_indices` — per-value index map implementing `scipy.ndimage.value_indices` (added in scipy 1.10.0). For each distinct voxel value, returns the row-major list of multi-indices `[i_0, …, i_{D-1}]` where it occurs. `ignore_value` keyword parameter mirrors scipy's. Generic over `B: Backend, const D: usize`; one authoritative implementation serves 1-D/2-D/3-D/arbitrary-D. `F32Key` newtype provides bit-equality + bit-hash for `f32` HashMap keys. 16 differential tests cross-validated against scipy v1.17.1. Plus incidental typo fix: `NyulUdapaNormalizer` → `NyulUdupaNormalizer` in `statistics/mod.rs`. 1521/0/1 ritk-core tests.
- **Sprint 337** (v0.51.5, `ritk-core` 0.5.0): Morphological Laplacian filter — implements `scipy.ndimage.morphological_laplace` (D + E − 2f) with half-sample symmetric reflect-mode boundary handling (period 2n). Cubic structuring element of half-width `radius`. 9 differential tests cross-validated against scipy v1.17.1, including full 64-voxel byte-exact match on 4×4×4 two-corner-voxels. Plus structural partition: `morphological_laplace.rs` (595 → 2 files). 1505/0/1 ritk-core tests.
- **Sprint 336** (v0.51.4, `ritk-core` 0.4.0): Chamfer distance transform — implements `scipy.ndimage.distance_transform_cdt` (Chessboard L∞ + Taxicab L1) with two 3×3×3 raster scans over the 7-tap predecessor + 7-tap successor half-mask covering all 26 unique neighbours. Interior distance convention (bg=0, fg=chamfer distance, all-fg=−1.0 sentinel). Anisotropic spacing extension. 18 differential tests cross-validated against scipy v1.17.1. Plus structural partitions: `rank.rs` (567→4 files) and `chamfer.rs` (673→4 files). 1496/0/1 ritk-core tests.
- **Sprint 335** (v0.51.1, `ritk-core` 0.3.0): Prewitt filter (3-D, separable, factor 18·h, replicate padding) + position-of-extrema (`maximum_position` / `minimum_position`, generic over B and D) + histogram with [min,max] range and bins. 1478/0/1 ritk-core tests.
- **Sprint 334** (v0.51.0, `ritk-core` 0.2.0): Morphology foundation (Offset3D, StructuringElement, sealed SeShape trait, Cube/Cross/Ball ZSTs) + percentile filter + rank filter with O(n) `select_nth_unstable_by` introselect. 1431/0/1 ritk-core tests.
- **Sprint 332** (v0.50.95): Documentation audit, compaction, and cleanup — 4 stale files deleted, ARCHIVE.md created (18k lines), 3 root files compacted (18k→~400 lines). Structural audit — 3 violations partitioned into directory modules; ZERO files > 500 lines workspace-wide.

- **Sprint 331** (v0.50.94): Clippy zero-warning — 28 + 110+ residual warnings eliminated across all 14 crates. 8 preemptive structural partitions (association, dimse, dicom, test modules). Flaky test hardening (`translation_recovery_shifted_gaussian`). Documentation overhaul (IMPLEMENTATION_SUMMARY.md, OPTIMIZATION.md, README.md).

- **Sprint 330** (v0.50.93): Architectural decomposition — Monolithic `types.rs` → `types/` vertical hierarchy (half_width, stack_weights, bin_range, parzen_config). Monolithic `sample.rs` → `sample/` vertical hierarchy (sample_window, sparse_entry). `ParzenConfig::half_width()` and `inv_2sigma_sq()` promoted to production API. Computation functions extracted into `accumulate.rs`, `compute_direct.rs`, `compute_sparse.rs`. 24 new tests, 547 total ritk-registration tests.

- **Sprint 329** (v0.50.92): Sparse full joint normalization — `inv_sum_f` stored per-sample in `SparseWFixedT`, making direct↔sparse histograms numerically identical. FMA-idiomatic inner accumulation loop retained. Structural size regression tests added. 24 new tests, 523 total ritk-registration tests.

- **Sprint 328** (v0.50.91): Per-sample weight normalization — `accumulate_sample_direct` multiplies by `inv_sum_f × inv_sum_m` (σ²-invariant histogram total). Sparse-path moving-axis normalization. `StackWeights::len()`/`BinRange::len()` promoted to production. 18 new tests, 499 total ritk-registration tests.

## Testing

```bash
# Native test suite through the committed time budgets
cargo nextest run --workspace --lib --tests

# Documentation tests
cargo test --workspace --doc

# Focused package examples
cargo nextest run -p ritk-core
cargo nextest run -p ritk-io
cargo nextest run -p ritk-registration
cargo nextest run -p ritk-model
```

## Future Work

- [ ] Sinc interpolation
- [x] MINC format reader/writer (via [consus](https://github.com/ryancinsight/consus) pure-Rust HDF5)
- [x] Analyze format reader/writer
- [ ] Diffeomorphic Demons exact inverse
- [x] Curvature anisotropic diffusion (Alvarez et al. 1992)
- [x] Sato line / Hessian blob detection (Sato 1998)
- [x] Confidence / neighborhood connected region growing
- [x] Skeletonization (hole filling remains)
- [ ] Longitudinal analysis pipeline
- [ ] WGSL/compute-shader kernels for critical filters

- [x] ONNX model import for DL registration (RITK 0.20.1+, `onnx-ir` parsing, initializers, graph validation)
- [ ] Expand Python bindings to cover model inference
- [ ] Publish to crates.io and PyPI

## License

RITK is dual-licensed under either of the following, at your option:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

## Contributing

Contributions are welcome. Requirements:

- Follow the existing deep-hierarchy architecture (DIP, SSOT, SoC, SRP)
- Include tests with analytically derived expected values
- Update documentation alongside implementation changes
- No namespace bleeding or unnecessary wrapper types

## Acknowledgments

- Inspired by [ITK](https://itk.org/) (Insight Segmentation and Registration Toolkit)
- Uses Coeus and Leto for tensor and numerical execution
- Uses [nalgebra](https://nalgebra.org/) for linear algebra
