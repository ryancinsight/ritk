# RITK — Rust Image Toolkit

A high-performance medical image processing and registration toolkit built in Rust, inspired by ITK concepts with GPU acceleration through Burn.

## Overview

RITK provides a comprehensive framework for medical image analysis:

- **GPU Acceleration**: Built on the Burn framework for efficient GPU/CPU tensor computation with automatic differentiation
- **Deep Module Hierarchy**: Strict DIP/SSOT/SoC/SRP architecture across workspace crates
- **Broad Format Support**: DICOM, NIfTI, MetaImage, NRRD, PNG, TIFF/BigTIFF, MGH/MGZ, VTK, JPEG
- **Classical & Deformable Registration**: Rigid, affine, B-Spline FFD, Demons, SyN, LDDMM, Atlas/Groupwise
- **Deep-Learning Registration**: TransMorph, SSMMorph via Burn autodiff
- **Image Processing Pipeline**: Filtering, segmentation, statistics, normalization
- **CT Visualization Support**: Bed separation filter for CT foreground/body masking
- **Native DICOM Viewer**: `ritk-snap` desktop viewer with DICOM folder/DICOMDIR launch, MPR viewports, overlays, measurements, and tag inspection
- **Python Bindings**: PyO3 + maturin with NumPy bridge, packaged type stubs, and `py.typed`
- **CLI**: `ritk` binary with `convert`, `filter`, `register`, `segment`, and `stats` subcommands

## Crate Structure

```
ritk/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── ritk-core/                # Core types, filters, segmentation, statistics
│   │   └── src/
│   │       ├── spatial/          # Point, Vector, Spacing, Direction (nalgebra)
│   │       ├── image/            # Image<B,D> with physical metadata
│   │       ├── transform/        # Transform trait + implementations
│   │       ├── interpolation/    # Interpolator trait + implementations
│   │       ├── filter/           # Image filters
│   │       ├── segmentation/     # Segmentation algorithms + distance transform
│   │       └── statistics/       # Image statistics, normalization, comparison
│   ├── ritk-dicom/               # DICOM transfer syntax, pixel codec contracts, backend boundary
│   │   └── src/
│   │       ├── syntax/           # Canonical TransferSyntaxKind and predicates
│   │       ├── pixel/            # PixelLayout and modality LUT byte decode
│   │       ├── codec/native/     # Native PackBits, RLE Lossless, and JPEG grayscale decoders
│   │       └── backend/          # NativeCodecBackend, FrameDecodeBackend, and dicom-rs fallback adapter
│   ├── ritk-io/                  # Format readers/writers
│   │   └── src/format/
│   │       ├── dicom/
│   │       ├── nifti/
│   │       ├── metaimage/        # .mha/.mhd
│   │       ├── nrrd/
│   │       ├── png/
│   │       ├── tiff/             # TIFF/BigTIFF
│   │       ├── mgh/              # MGH/MGZ (FreeSurfer)
│   │       ├── vtk/              # VTK legacy structured points
│   │       └── jpeg/             # JPEG 2D grayscale
│   ├── ritk-registration/        # Registration framework
│   │   └── src/
│   │       ├── metric/           # Similarity metrics
│   │       ├── optimizer/        # Optimization algorithms
│   │       ├── regularization/   # Deformation regularizers
│   │       ├── classical/        # Kabsch SVD, MI-based rigid/affine
│   │       ├── demons/           # Thirion, Diffeomorphic, Symmetric
│   │       ├── diffeomorphic/    # Greedy SyN, Multi-Resolution SyN, BSpline SyN
│   │       ├── bspline_ffd/      # BSpline free-form deformation
│   │       ├── lddmm/           # Large Deformation Diffeomorphic Metric Mapping
│   │       ├── atlas/            # Groupwise atlas registration + label fusion
│   │       ├── registration/     # DL registration losses, SSM registration
│   │       ├── validation/       # Registration quality assessment
│   │       └── progress/         # Progress reporting
│   ├── ritk-model/               # Deep-learning registration models
│   │   └── src/
│   │       ├── transmorph/       # TransMorph architecture
│   │       ├── ssmmorph/         # SSMMorph architecture
│   │       ├── affine/           # Learned affine alignment
│   │       └── io/               # Model I/O
│   ├── ritk-python/              # Python bindings (PyO3 + maturin)
│   │   └── src/
│   │       ├── filter.rs         # 34 filter functions
│   │       ├── segmentation.rs   # 27 segmentation functions
│   │       ├── registration.rs   # 13 registration / atlas / label-fusion functions
│   │       ├── statistics.rs     # 13 statistics/normalization/comparison functions
│   │       ├── image.rs          # Image wrapper
│   │       └── io.rs             # Image + composite transform I/O
│   ├── ritk-snap/                # Native DICOM/NIfTI viewer
│   │   └── src/
│   │       ├── dicom/            # Input path normalization, series tree, loader bridge, metadata tag table
│   │       ├── label/            # Segmentation label editor over ritk-core annotation primitives
│   │       ├── render/           # Slice extraction, window/level, colormap LUTs
│   │       ├── session/          # Serializable viewer session snapshots
│   │       ├── tools/            # Measurement and interaction state
│   │       └── ui/               # egui layout, linked MPR cursor, viewport, sidebar, toolbar, overlays
│   └── ritk-cli/                 # CLI binary
│       └── src/commands/
│           ├── convert.rs
│           ├── filter.rs
│           ├── register.rs
│           ├── segment.rs
│           └── stats.rs
```

### Viewer (`ritk-snap`)

`ritk-snap [PATH]` launches the native viewer directly against a DICOM folder,
a `DICOMDIR` file, or a supported medical image file. The viewer keeps DICOM
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
W/L drag mapping (horizontal width, vertical center) via SSOT `ui/window_level`,
PNG slice export, full axial/coronal/sagittal MPR PNG export, DICOM overlays,
RT-STRUCT contour overlay loading and rendering,
and a
deterministic Tags panel covering series metadata, first-slice geometry/display
tags, private scalar tags, preserved object-model nodes, and raw preserved
element byte counts. Viewer session save/load stores presentation state as
JSON, including source path, slice indices, window/level, colormap, active
tool, layout flags, overlay flags, sidebar tab, pan, and zoom.

## Features

### Core Types (`ritk-core`)

**Spatial types** — `Point<D>`, `Vector<D>`, `Spacing<D>`, `Direction<D>` backed by nalgebra.

**Image** — `Image<B, D>` carrying tensor data plus physical metadata (origin, spacing, direction) with index↔physical coordinate transforms.

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

- **14 filter functions** (Gaussian, median, bilateral, Canny, Frangi, N4, etc.)
- **16 segmentation functions** (Otsu family, morphology, connected components, watershed, level sets, etc.)
- **11 registration functions** (Thirion/Diffeomorphic/Symmetric Demons, SyN, Multi-Res SyN, BSpline SyN, BSpline FFD, LDDMM, atlas building, majority-vote fusion, joint label fusion)
- **13 statistics functions** (descriptive stats, Dice, Hausdorff, PSNR, SSIM, noise estimation, normalization, white stripe)
- NumPy ↔ `Image` zero-copy bridge
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
```
```

## Usage Example

```rust
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::transform::RigidTransform;
use ritk_registration::metric::MeanSquaredError;
use ritk_registration::optimizer::GradientDescent;
use ritk_registration::registration::Registration;
use ritk_io::read_nifti;

type Backend = Autodiff<NdArray<f32>>;

// Load images
let device = Default::default();
let fixed: Image<Backend, 3> = read_nifti("fixed.nii", &device)?;
let moving: Image<Backend, 3> = read_nifti("moving.nii", &device)?;

// Set up registration components
let transform = RigidTransform::<Backend, 3>::identity(None, &device);
let metric = MeanSquaredError::new();
let optimizer = GradientDescent::new(0.01);

// Run registration
let mut registration = Registration::new(optimizer, metric);
let result = registration.execute(&fixed, &moving, transform, 100, 0.01);
```

## Dependencies

| Crate | Role |
|---|---|
| `burn` | GPU/CPU tensor ops with autodiff |
| `nalgebra` | Linear algebra, spatial types |
| `dicom` | DICOM format support |
| `nifti` | NIfTI format support |
| `rayon` | CPU parallelism |
| `pyo3` / `numpy` | Python bindings |
| `serde` | Serialization (transform I/O) |
| `anyhow` / `thiserror` | Error handling |
| `tracing` | Structured logging |

## Building

```bash
# Build all crates (release)
cargo build --release

# Run all tests
cargo test --all

# Build Python extension / install into current environment
cd crates/ritk-python && maturin develop --release

# Install CLI
cargo install --path crates/ritk-cli
```

## Testing

```bash
# Full test suite
cargo test --all

# Per-crate
cargo test -p ritk-core
cargo test -p ritk-io
cargo test -p ritk-registration
cargo test -p ritk-model
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

See workspace `Cargo.toml` or individual crate manifests for license terms.

## Contributing

Contributions are welcome. Requirements:

- Follow the existing deep-hierarchy architecture (DIP, SSOT, SoC, SRP)
- Include tests with analytically derived expected values
- Update documentation alongside implementation changes
- No namespace bleeding or unnecessary wrapper types

## Acknowledgments

- Inspired by [ITK](https://itk.org/) (Insight Segmentation and Registration Toolkit)
- Built on [Burn](https://burn.dev/) for GPU-accelerated tensor computation
- Uses [nalgebra](https://nalgebra.org/) for linear algebra
