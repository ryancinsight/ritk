# ritk-transform

Spatial image transform operators for [RITK](https://github.com/ryancinsight/ritk).

Concrete implementations of the `Transform` and `Resampleable` traits.

| Transform | Description |
|---|---|
| `Translation` | Pure translation |
| `Rigid` | Rotation + translation |
| `Affine` | Full affine (12 DOF in 3-D) |
| `Scale` | Axis-aligned scaling |
| `Versor` | Unit-quaternion rotation (3-D) |
| `BSpline` | Free-form deformation on a control-point lattice |
| `DisplacementField` | Dense voxel-wise displacement |
| `ChainedTransform` | Sequential composition |
| `CompositeTransform` | Named composite with JSON serialization |

All transforms are generic over the Coeus backend and dimension; none is cloned
per scalar type or dimensionality.

## Usage

```toml
[dependencies]
ritk-transform = "0.2.0"
```
