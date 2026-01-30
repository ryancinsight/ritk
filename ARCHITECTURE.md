# RITK Architecture

## Design Philosophy

RITK follows a strict architectural philosophy based on three core principles:

### 1. Single Source of Truth (SSOT)

Each type and operation has exactly one authoritative definition in the codebase. This eliminates:
- Duplicate type definitions
- Inconsistent implementations
- Confusion about which version to use

**Example**: The `Point<D>` type is defined once in `ritk-core/src/spatial/point.rs` and re-exported through the module hierarchy. There are no wrapper types or aliases that hide the underlying nalgebra type.

### 2. Single Responsibility Principle (SRP)

Each module and type has one clear, focused purpose:

- **`spatial/`**: Pure spatial types (Point, Vector, Spacing, Direction)
- **`image/`**: Image data and metadata only
- **`transform/`**: Spatial transformations only
- **`interpolation/`**: Interpolation algorithms only
- **`metric/`**: Similarity metrics only
- **`optimizer/`**: Optimization algorithms only

No module mixes concerns. For example, the `Image` type does not contain transformation logic - that's in the `transform` module.

### 3. Separation of Concerns (SOC)

Related functionality is grouped together, unrelated functionality is separated:

**Spatial Types** (`ritk-core/src/spatial/`):
- `point.rs`: Point coordinates
- `vector.rs`: Vector operations
- `spacing.rs`: Pixel/voxel spacing
- `direction.rs`: Orientation matrices

**Image Operations** (`ritk-core/src/image/`):
- `image.rs`: Image data structure
- `metadata.rs`: Physical metadata (origin, spacing, direction)

**Transformations** (`ritk-core/src/transform/`):
- `trait_.rs`: Transform trait definition
- `translation.rs`: Translation transform
- `rigid.rs`: Rigid transform (rotation + translation)
- `affine.rs`: Affine transform
- `bspline.rs`: B-Spline free-form deformation

**Interpolation** (`ritk-core/src/interpolation/`):
- `trait_.rs`: Interpolator trait
- `linear.rs`: Linear interpolation
- `nearest.rs`: Nearest neighbor interpolation

## Module Organization

### Deep Vertical Hierarchy

The project uses a deep vertical hierarchy where each domain has its own subdirectory:

```
ritk-core/src/
├── spatial/          # Spatial types
│   ├── mod.rs         # Public API
│   ├── point.rs       # Point type
│   ├── vector.rs      # Vector type
│   ├── spacing.rs     # Spacing type
│   └── direction.rs   # Direction matrix
├── image/            # Image types
│   ├── mod.rs         # Public API
│   ├── image.rs       # Image struct
│   └── metadata.rs    # Image metadata
├── transform/         # Transforms
│   ├── mod.rs         # Public API
│   ├── trait_.rs      # Transform trait
│   ├── translation.rs  # Translation
│   ├── rigid.rs       # Rigid transform
│   ├── affine.rs      # Affine transform
│   └── bspline.rs     # B-Spline transform
└── interpolation/     # Interpolation
    ├── mod.rs         # Public API
    ├── trait_.rs      # Interpolator trait
    ├── linear.rs      # Linear interpolation
    └── nearest.rs     # Nearest neighbor
```

### Shared Accessors

Each module's `mod.rs` file serves as the shared accessor, providing:
- Public API exports
- Type aliases for common dimensions (2D, 3D)
- Documentation for the module

**Example** (`ritk-core/src/spatial/mod.rs`):
```rust
pub mod point;
pub mod vector;
pub mod spacing;
pub mod direction;

pub use point::Point;
pub use vector::Vector;
pub use spacing::Spacing;
pub use direction::Direction;

// Common type aliases
pub type Point2 = Point<2>;
pub type Point3 = Point<3>;
pub type Vector2 = Vector<2>;
pub type Vector3 = Vector<3>;
pub type Spacing2 = Spacing<2>;
pub type Spacing3 = Spacing<3>;
pub type Direction2 = Direction<2>;
pub type Direction3 = Direction<3>;
```

## Domain-Level Naming

Types are named by their domain, not by their implementation:

✅ **Good**:
- `Point<D>` - Clear domain name
- `Vector<D>` - Clear domain name
- `Spacing<D>` - Clear domain name
- `Direction<D>` - Clear domain name

❌ **Bad** (avoided):
- `NalgebraPoint<D>` - Implementation detail
- `SpatialVector<D>` - Redundant
- `ImageSpacing<D>` - Redundant
- `OrientationMatrix<D>` - Implementation detail

## No Namespace Bleeding

Each crate maintains its own namespace without polluting the global namespace:

- `ritk-core` exports only core types
- `ritk-io` exports only I/O functions
- `ritk-registration` exports only registration types

Users explicitly import what they need:
```rust
use ritk_core::{Image, Point3, Spacing3};
use ritk_core::transform::RigidTransform;
use ritk_registration::{Registration, MeanSquaredError};
use ritk_io::read_nifti;
```

## No Excess Wrappers

We avoid unnecessary wrapper types and use underlying libraries directly:

✅ **Good**:
```rust
pub type Point<const D: usize> = NaPoint<f64, D>;
pub type Vector<const D: usize> = SVector<f64, D>;
```

❌ **Bad** (avoided):
```rust
pub struct Point<const D: usize> {
    inner: NaPoint<f64, D>,
}
impl<const D: usize> Point<D> {
    pub fn x(&self) -> f64 { self.inner.x }
    // ... many wrapper methods
}
```

The type alias approach:
- Provides all nalgebra functionality automatically
- No maintenance burden
- Clear and simple
- Zero runtime overhead

## Backend Abstraction

Burn provides the backend abstraction for GPU/CPU:

```rust
pub struct Image<B: Backend, const D: usize> {
    data: Tensor<B, D>,
    metadata: ImageMetadata<D>,
}
```

This allows:
- CPU backend for testing (`NdArray`)
- GPU backend for production (`Wgpu`)
- Future backends without code changes

## Coordinate Systems

RITK maintains clear separation between coordinate systems:

### Index Space
- Discrete pixel/voxel indices
- Integer coordinates
- Used for array indexing

### Physical Space
- Continuous coordinates in mm (or other units)
- Floating-point coordinates
- Used for spatial transformations

### Transformation
```rust
// Index → Physical
physical = origin + direction @ (index * spacing)

// Physical → Index
index = direction^(-1) @ (physical - origin) / spacing
```

## Error Handling

RITK uses `thiserror` for structured error types and `anyhow` for error propagation:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ImageError {
    #[error("Invalid image dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Failed to read file: {0}")]
    IoError(#[from] std::io::Error),
}
```

## Testing Strategy

Each module includes comprehensive tests:

- Unit tests for individual functions
- Integration tests for module interactions
- Property-based tests where applicable

**Example** (`ritk-core/src/spatial/point.rs`):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let p = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
    }
}
```

## Future Extensions

The architecture supports easy extension:

### Adding a New Transform
1. Create `ritk-core/src/transform/new_transform.rs`
2. Implement `Transform<B, D>` trait
3. Add to `ritk-core/src/transform/mod.rs`
4. Add tests

### Adding a New Metric
1. Create `ritk-registration/src/metric/new_metric.rs`
2. Implement `Metric<B, D>` trait
3. Add to `ritk-registration/src/metric/mod.rs`
4. Add tests

### Adding a New Interpolator
1. Create `ritk-core/src/interpolation/new_interpolator.rs`
2. Implement `Interpolator<B>` trait
3. Add to `ritk-core/src/interpolation/mod.rs`
4. Add tests

## Performance Considerations

### GPU Acceleration
- All tensor operations use Burn for GPU acceleration
- Batch operations preferred over loops
- Memory layout optimized for GPU access patterns

### Zero-Copy Operations
- Type aliases avoid wrapper overhead
- Direct use of nalgebra types
- Efficient tensor operations via Burn

### Compile-Time Optimization
- Const generics for dimensionality
- Monomorphization for specialized code
- Inlining where beneficial

## Conclusion

This architecture provides:
- **Clarity**: Easy to understand and navigate
- **Maintainability**: Changes are localized
- **Extensibility**: New features fit naturally
- **Performance**: No unnecessary overhead
- **Correctness**: Single source of truth for each type

The deep vertical hierarchy with shared accessors ensures that:
- Each domain is self-contained
- Dependencies are explicit
- Testing is straightforward
- Documentation is organized
