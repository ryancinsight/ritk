# atlas/RITK: Medical Image Processing and Registration

Redundancy-free Coeus-native medical image processing, registration, and analysis.

## Overview

RITK is the atlas medical image processing and registration toolkit. It provides
Coeus-native implementations of ITK-style image filters, registration metrics,
and format I/O, all built on the atlas foundation crates (leto, eunomia,
hermes, moirai, coeus).

## Architecture

The workspace follows the atlas deep vertical hierarchy with strict
Separation of Concerns:

```
ritk/
├── crates/
│   ├── ritk-core/          # Image data structures and spatial primitives
│   ├── ritk-image/         # Image type and boundary traits
│   ├── ritk-filter/        # Image filter implementations (intensity, morphology, diffusion)
│   ├── ritk-registration/  # Registration metrics and optimization
│   ├── ritk-io/            # Format readers/writers (DICOM, NIfTI, NRRD, PNG, JPEG, VTK)
│   ├── ritk-statistics/    # Image comparison metrics (Dice, PSNR, SSIM)
│   ├── ritk-transform/     # Coordinate transformations
│   ├── ritk-interpolation/ # Resampling and interpolation
│   ├── ritk-morphology/    # Morphological operations
│   ├── ritk-snap/          # 3D mesh visualization
│   └── ritk-python/        # PyO3 bindings for Python interop
├── docs/
│   ├── book/               # mdBook documentation (this file)
│   ├── adr/               # Architecture Decision Records
│   └── atlas-migration/    # Migration documentation
└── xtask/                 # Build utilities and CI helpers
```

## Key Design Principles

- **Zero-cost abstractions**: All filter implementations compile to machine code
  identical to hand-written concrete specializations via monomorphization.
- **Zero-copy I/O**: Deserialize/parse into borrowed views over source buffers;
  use Coeus tensor views instead of owned copies where possible.
- **Redundancy-free**: One canonical implementation per operation family; no
  cloned algorithm variants across scalar types or backends.
- **SSOT boundaries**: Each format owns its file-axis contract in a dedicated
  `spatial.rs` module.

## Build and Test

```bash
# Build all crates
cargo check -p ritk-filter --lib

# Run tests
cargo test -p ritk-filter --lib

# Build docs
mdbook build docs/book
```

## References

- [atlas Migration Summary](coeus_migration.md)
- [atlas Architecture Decision Records](adr/)
