# RITK - Rust Image Toolkit

A high-performance medical image registration toolkit built with Rust, leveraging ITK concepts and modern GPU acceleration through Burn.

## Overview

RITK provides a comprehensive framework for medical image registration with:
- **GPU Acceleration**: Built on Burn framework for efficient GPU computation
- **Modern Architecture**: Deep vertical hierarchical structure with clear separation of concerns
- **Medical Image Support**: DICOM and NIfTI file formats via dicom-rs and nifti-rs
- **Flexible Transforms**: Translation, Rigid, Affine, and B-Spline transforms
- **Multiple Metrics**: Mean Squared Error, Mutual Information, Normalized Cross-Correlation
- **Autodiff Ready**: Built-in support for automatic differentiation and AI model training

## Architecture

### Design Principles

1. **Single Source of Truth (SSOT)**: Each type and operation has a single, authoritative definition
2. **Single Responsibility Principle (SRP)**: Each module has a clear, focused purpose
3. **Separation of Concerns (SOC)**: Spatial types, transforms, interpolation, and I/O are cleanly separated
4. **Domain-Level Naming**: No namespace bleeding - types are named by their domain (e.g., `Point`, `Vector`, `Spacing`)
5. **No Excess Wrappers**: Direct use of underlying libraries (nalgebra, burn) without unnecessary abstraction layers

### Crate Structure

```
ritk/
├── Cargo.toml                 # Workspace configuration
├── crates/
│   ├── ritk-core/            # Core types and operations
│   │   ├── src/
│   │   │   ├── spatial/      # Spatial types (Point, Vector, Spacing, Direction)
│   │   │   │   ├── mod.rs
│   │   │   │   ├── point.rs
│   │   │   │   ├── vector.rs
│   │   │   │   ├── spacing.rs
│   │   │   │   └── direction.rs
│   │   │   ├── image/        # Image types and metadata
│   │   │   │   ├── mod.rs
│   │   │   │   ├── image.rs
│   │   │   │   └── metadata.rs
│   │   │   ├── transform/    # Transform types
│   │   │   │   ├── mod.rs
│   │   │   │   ├── trait_.rs
│   │   │   │   ├── translation.rs
│   │   │   │   ├── rigid.rs
│   │   │   │   ├── affine.rs
│   │   │   │   └── bspline.rs
│   │   │   ├── interpolation/ # Interpolation methods
│   │   │   │   ├── mod.rs
│   │   │   │   ├── trait_.rs
│   │   │   │   ├── linear.rs
│   │   │   │   └── nearest.rs
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   ├── ritk-io/              # I/O operations
│   │   ├── src/
│   │   │   ├── nifti/       # NIfTI file I/O
│   │   │   │   ├── mod.rs
│   │   │   │   ├── reader.rs
│   │   │   │   └── writer.rs
│   │   │   ├── dicom/       # DICOM file I/O
│   │   │   │   ├── mod.rs
│   │   │   │   ├── reader.rs
│   │   │   │   └── writer.rs
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   └── ritk-registration/     # Registration framework
│       ├── src/
│       │   ├── metric/       # Similarity metrics
│       │   │   ├── mod.rs
│       │   │   ├── trait_.rs
│       │   │   ├── mse.rs
│       │   │   ├── mutual_information.rs
│       │   │   └── ncc.rs
│       │   ├── optimizer/     # Optimizers
│       │   │   ├── mod.rs
│       │   │   ├── trait_.rs
│       │   │   └── gradient_descent.rs
│       │   ├── registration.rs
│       │   └── lib.rs
│       └── Cargo.toml
```

## Core Types

### Spatial Types (`ritk-core::spatial`)

- **`Point<D>`**: Physical coordinates (based on nalgebra)
- **`Vector<D>`**: Spatial displacements and directions
- **`Spacing<D>`**: Physical distance between pixels/voxels
- **`Direction<D>`**: Image orientation matrix

### Image Type (`ritk-core::image`)

- **`Image<B, D>`**: Medical image with physical metadata
  - Tensor data on GPU/CPU
  - Origin, spacing, direction metadata
  - Coordinate transformations (index ↔ physical space)

### Transforms (`ritk-core::transform`)

- **`Transform<B, D>`**: Core transform trait
- **`TranslationTransform<B, D>`**: Simple translation
- **`RigidTransform<B, D>`**: Rotation + translation
- **`AffineTransform<B, D>`**: General affine transformation
- **`BSplineTransform<B, D>`**: Free-form deformation

### Interpolation (`ritk-core::interpolation`)

- **`Interpolator<B>`**: Core interpolator trait
- **`LinearInterpolator`**: Bilinear/trilinear interpolation
- **`NearestNeighborInterpolator`**: Nearest neighbor interpolation

## Registration Framework (`ritk-registration`)

### Metrics

- **`Metric<B, D>`**: Core metric trait
- **`MeanSquaredError`**: MSE similarity metric
- **`MutualInformation`**: MI similarity metric (TODO)
- **`NormalizedCrossCorrelation`**: NCC similarity metric (TODO)

### Optimizers

- **`Optimizer<M, B>`**: Core optimizer trait
- **`GradientDescentOptimizer`**: Simple gradient descent with momentum

### Registration

- **`Registration<B, O, M, T, D>`**: Main registration framework
  - Combines metric, optimizer, and transform
  - Iterative optimization loop
  - Automatic differentiation support

## I/O Operations (`ritk-io`)

### NIfTI Support

- **`read_nifti<B, P>`**: Read NIfTI files
- **`write_nifti<B, P>`**: Write NIfTI files (TODO)

### DICOM Support

- **`read_dicom_series<B, P>`**: Read DICOM series
- **`write_dicom<B, P>`**: Write DICOM files (TODO)

## Usage Example

```rust
use ritk_core::{Image, Point3, Spacing3, Direction3};
use ritk_core::transform::RigidTransform;
use ritk_registration::{Registration, MeanSquaredError, GradientDescentOptimizer};
use ritk_io::read_nifti;
use burn_ndarray::NdArray;

type Backend = NdArray<f32>;

// Load images
let fixed = read_nifti::<Backend, _>("fixed.nii", &device)?;
let moving = read_nifti::<Backend, _>("moving.nii", &device)?;

// Create transform
let translation = Tensor::zeros([3], &device);
let rotation = Tensor::zeros([3], &device);
let transform = RigidTransform::new(translation, rotation);

// Create metric and optimizer
let metric = MeanSquaredError::new();
let optimizer = GradientDescentOptimizer::default_params();

// Run registration
let mut registration = Registration::new(optimizer, metric);
let result = registration.execute(
    &fixed,
    &moving,
    transform,
    100,  // iterations
    0.01, // learning rate
);
```

## Dependencies

- **burn**: GPU/CPU tensor operations with autodiff
- **nalgebra**: Linear algebra operations
- **dicom**: DICOM file format support
- **nifti**: NIfTI file format support
- **thiserror**: Error handling
- **anyhow**: Error propagation
- **tracing**: Logging and instrumentation

## Building

```bash
# Build all crates
cargo build --release

# Run tests
cargo test --all

# Run with GPU support (requires wgpu feature)
cargo build --features wgpu
```

## Testing

```bash
# Run all tests
cargo test --all

# Run tests for specific crate
cargo test -p ritk-core
cargo test -p ritk-io
cargo test -p ritk-registration
```

## Future Work

- [ ] Complete B-Spline transform implementation
- [ ] Implement Mutual Information metric
- [ ] Implement Normalized Cross-Correlation metric
- [ ] Add multi-resolution registration
- [ ] Implement more optimizers (Adam, L-BFGS)
- [ ] Add resampling utilities
- [ ] Implement DICOM writing
- [ ] Add more interpolation methods (cubic, sinc)
- [ ] Create example applications
- [ ] Add Python bindings via PyO3

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing architecture
- Tests are included for new features
- Documentation is updated
- No namespace bleeding or excess wrappers

## Acknowledgments

- Inspired by ITK (Insight Segmentation and Registration Toolkit)
- Built with Burn framework for GPU acceleration
- Uses nalgebra for efficient linear algebra
