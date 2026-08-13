# ritk-core

Public facade contracts for [RITK](https://github.com/ryancinsight/ritk).

`ritk-core` re-exports the domain vocabulary so a consumer can depend on one
crate for the core types, and owns the trait bounds the format crates implement
against.

## Re-exports

| Module | From | Items |
|---|---|---|
| `spatial` | `ritk-spatial` | `Point`, `Vector`, `Spacing`, `Direction`, `VoxelIndex` |
| `image` | `ritk-image` | `Image`, `RgbVolume`, `ColorVolume` |
| `transform` | `ritk-transform` | Affine, rigid, versor, B-spline, displacement-field transforms |
| `interpolation` | `ritk-interpolation` | Linear, nearest-neighbor, B-spline interpolators |
| `io_bounds` | — | Trait bounds shared by the format crates |

Algorithms live in the operation crates (`ritk-filter`, `ritk-segmentation`,
`ritk-statistics`, `ritk-registration`), not here.

## Usage

```toml
[dependencies]
ritk-core = "0.10.0"
```
