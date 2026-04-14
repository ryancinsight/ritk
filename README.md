# RITK — Rust Image Toolkit

A high-performance medical image processing and registration toolkit built in Rust, inspired by ITK concepts with GPU acceleration through Burn.

## Overview

RITK provides a comprehensive framework for medical image analysis:

- **GPU Acceleration**: Built on the Burn framework for efficient GPU/CPU tensor computation with automatic differentiation
- **Deep Module Hierarchy**: Strict DIP/SSOT/SoC/SRP architecture across six workspace crates
- **Broad Format Support**: DICOM, NIfTI, MetaImage, NRRD, PNG, TIFF/BigTIFF
- **Classical & Deformable Registration**: Rigid, affine, B-Spline FFD, Demons, SyN, LDDMM
- **Deep-Learning Registration**: TransMorph, SSMMorph via Burn autodiff
- **Image Processing Pipeline**: Filtering, segmentation, statistics, normalization
- **Python Bindings**: PyO3 + maturin with NumPy bridge
- **CLI**: `ritk` binary with `convert`, `filter`, `register`, and `segment` subcommands

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
│   │       ├── segmentation/     # Segmentation algorithms
│   │       └── statistics/       # Image statistics & normalization
│   ├── ritk-io/                  # Format readers/writers
│   │   └── src/format/
│   │       ├── dicom/
│   │       ├── nifti/
│   │       ├── metaimage/        # .mha/.mhd
│   │       ├── nrrd/
│   │       ├── png/
│   │       └── tiff/             # TIFF/BigTIFF
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
│   │       ├── filter.rs         # 14 filter functions
│   │       ├── segmentation.rs   # 16 segmentation functions
│   │       ├── registration.rs   # 8 registration functions
│   │       ├── image.rs          # Image wrapper
│   │       └── io.rs             # Format I/O
│   └── ritk-cli/                 # CLI binary
│       └── src/commands/
│           ├── convert.rs
│           ├── filter.rs
│           ├── register.rs
│           └── segment.rs
```

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

**Segmentation**

| Category | Algorithms |
|---|---|
| Thresholding | Otsu, Multi-Otsu, Li, Yen, Kapur, Triangle |
| Binary Morphology | Erosion, Dilation, Opening, Closing |
| Labeling | Connected Components (Hoshen–Kopelman) |
| Region Growing | Seeded region growing |
| Clustering | K-Means |
| Watershed | Marker-controlled watershed |
| Level Sets | Chan–Vese, Geodesic Active Contour |

**Statistics & Normalization**

| Category | Functions |
|---|---|
| Descriptive | Min, Max, Mean, Variance, Percentile (masked support) |
| Comparison | Dice, Hausdorff Distance, Mean Surface Distance, PSNR, SSIM |
| Normalization | Min-Max, Z-Score, Histogram Matching, Nyúl–Udupa Histogram Standardization |
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

### Deep-Learning Models (`ritk-model`)

| Model | Description |
|---|---|
| TransMorph | Transformer-based deformable registration |
| SSMMorph | Statistical shape model registration |

### Python Bindings (`ritk-python`)

PyO3 + maturin package exposing:

- **14 filter functions** (Gaussian, median, bilateral, Canny, Frangi, N4, etc.)
- **16 segmentation functions** (Otsu family, morphology, connected components, watershed, level sets, etc.)
- **8 registration functions** (Thirion/Diffeomorphic/Symmetric Demons, SyN, Multi-Res SyN, BSpline SyN, BSpline FFD, LDDMM)
- NumPy ↔ `Image` zero-copy bridge
- Format I/O for all supported formats

### CLI (`ritk-cli`)

```
ritk convert  <input> <output>          # Format conversion
ritk filter   <input> <output> [opts]   # Apply filters
ritk register <fixed> <moving> [opts]   # Run registration
ritk segment  <input> <output> [opts]   # Run segmentation
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

# Build Python wheel
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
- [ ] Atlas-based segmentation
- [ ] Groupwise registration
- [ ] Longitudinal analysis pipeline
- [ ] WGSL/compute-shader kernels for critical filters
- [ ] ONNX model import for DL registration
- [ ] Expand Python bindings to cover statistics and model inference
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