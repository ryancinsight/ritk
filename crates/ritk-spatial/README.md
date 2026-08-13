# ritk-spatial

Validated spatial geometry and coordinate transforms for [RITK](https://github.com/ryancinsight/ritk).

Leaf crate of the RITK workspace: no RITK-internal dependencies, only `leto` and
`serde`.

## Types

| Type | Description |
|---|---|
| `Point<D>` | Position in D-dimensional physical space |
| `Vector<D>` | Displacement in D-dimensional physical space |
| `Spacing<D>` | Per-axis voxel spacing |
| `Direction<D>` | Orthonormal direction cosine matrix |
| `VoxelIndex<D>` | Discrete voxel coordinate |
| `VolumeDims<D>` | Validated volume extent |
| `CoordinateMap` | Index-to-physical and physical-to-index mapping |

Each is a transparent newtype over a Leto stack-backed primitive
(`leto::FixedVector`, `leto::FixedMatrix`), so `Point` and `Vector` stay
distinct types to the compiler at zero runtime cost, and validation and
serialization live on the newtype rather than the provider type.

## Usage

```toml
[dependencies]
ritk-spatial = "0.2.0"
```
