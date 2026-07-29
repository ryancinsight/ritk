# RITK: Medical Image Processing and Registration

RITK is a Rust toolkit for loading medical images, preserving their physical
geometry, applying ITK-style filters, evaluating similarity, and producing
classical or differentiable registrations. The book is organized as a
user-facing workflow:

1. load a volume through ritk-io;
2. inspect shape, spacing, origin, and direction;
3. preprocess intensities and spatial structure;
4. choose a same-modality or multi-modal metric;
5. resample and validate the result; and
6. write the output without losing the spatial contract.

The examples use real RITK APIs and committed fixtures where a dataset is
needed. Synthetic examples are deterministic, small, and input-sensitive so
they can generate figures in CI.

## First runnable workflow

~~~rust,ignore
use coeus_core::SequentialBackend;
use ritk_filter::IntensityWindowingFilter;
use ritk_io::read_image_native;

let backend = SequentialBackend;
let input = read_image_native("volume.nii.gz")?;
let windowed = IntensityWindowingFilter::new(-160.0, 240.0, 0.0, 1.0)
    .apply_native(&input, &backend)?;
println!("shape = {:?}, spacing = {:?}", windowed.shape(), windowed.spacing());
~~~

The NativeImage alias is Image<f32, SequentialBackend, 3>. RITK stores voxels
in [depth, row, column] order while origin, spacing, and direction describe
the physical frame. Filters generally preserve that metadata; resampling and
transforms are the deliberate exceptions.

## Workspace map

| Crate | User-facing role |
| --- | --- |
| ritk-io | Format inference, native readers, native writers, and DICOM series handling |
| ritk-image | Coeus-backed image storage and physical-coordinate metadata |
| ritk-filter | Intensity, smoothing, edge, morphology, diffusion, and spatial filters |
| ritk-registration | Metrics, transforms, classical registration, and differentiable registration |
| ritk-statistics | Histogram, similarity, overlap, and image-quality statistics |
| ritk-transform / ritk-interpolation | Transform parameterization and physical resampling |

## Build the book and examples

~~~text
# Run tests with the repository's bounded native-test runner
cargo nextest run -p ritk-filter

# Build docs
mdbook build docs/book
mdbook test docs/book
cargo build -p ritk-filter --examples
cargo build -p ritk-io --examples
cargo build -p ritk-registration --examples
~~~

For native tests use the repository's configured cargo nextest command.
Doctests use cargo test --doc. A figure is valid only after its generating
example succeeds and the rendered artifact has been inspected.

## How to read the chapters

- Part I explains format boundaries and spatial-axis conventions.
- Part II builds filtering pipelines from intensity, spatial, morphology, and
  diffusion operations.
- Part III covers metrics, transforms, classical and differentiable
  registration, and post-registration validation.
- Part IV explains backend dispatch, zero-copy boundaries, and measurements.
- Part V maps the public RITK surface onto Coeus, Leto, and Moirai.

- [Provider migration summary](../coeus_migration.md)
- [Architecture decision records](../adr/README.md)
