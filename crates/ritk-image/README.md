# ritk-image

Native medical image storage and metadata for [RITK](https://github.com/ryancinsight/ritk).

Defines `Image<T, B, D>` — typed Coeus tensor storage carrying origin, spacing,
direction, and coordinate-map metadata, with index-to-physical and
physical-to-index transforms. `VoxelImage<B, D>` selects a supported stored
scalar type at runtime while retaining its values and geometry. Numeric
conversion policy and loss reporting belong to the `ritk-io` boundary.

## Types

| Type | Description |
|---|---|
| `Image<T, B, D>` | Scalar volume over a Coeus backend with physical metadata |
| `VoxelImage<B, D>` | Runtime-selected scalar image |
| `RgbVolume` / `ColorVolume` | Multi-channel color volumes |

Depends on `ritk-spatial` for spatial types and the Coeus tensor contracts for
storage. Carries no I/O, filtering, or registration logic.

## Usage

```toml
[dependencies]
ritk-image = "0.4.0"
ritk-spatial = "0.2.0"
```

```rust
use ritk_image::tensor::SequentialBackend;
use ritk_image::{Image, VoxelImage};
use ritk_spatial::{Direction, Point, Spacing};

let backend = SequentialBackend;
let image = Image::from_flat_on(
    vec![12_u16, 900],
    [2],
    Point::new([0.0]),
    Spacing::new([1.0]),
    Direction::identity(),
    &backend,
)
.expect("valid image dimensions");
let image = VoxelImage::Unsigned16(image);
assert!(matches!(image, VoxelImage::Unsigned16(_)));
```
