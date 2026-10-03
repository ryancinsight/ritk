# ritk-vtk

VTK-native data model and I/O for [RITK](https://github.com/ryancinsight/ritk).

Provides the authoritative VTK data model and the VTK-format read/write free
functions. Deliberately independent of the `ritk-io` domain traits so the VTK
domain carries no orphan-rule coupling; `ritk-io` adapts it for unified
dispatch.

Color mapping uses the [Iris](https://github.com/ryancinsight/iris)
`NamedColorMap` contract rather than a local interpolation path.

`VtkImageVolume` is the direction-aware, zero-copy handoff for regular image
volumes. It stores VTK-order dimensions, origin, spacing, direction, channel
count and shared scalar samples. `to_vtk_image_data` is an explicit copy
boundary for legacy serializers and filters whose attribute arrays own a
`Vec<f32>`.

The legacy structured-points reader maps VTK XYZ dimensions and spacing to
RITK's ZYX tensor axes while preserving physical origin in XYZ order. This
axis mapping lives in `ritk-vtk`: `VtkImageVolume::from_tensor_parts` accepts
RITK-order metadata and constructs the VTK-order handoff. Consumer crates pass
the metadata without format-specific reordering. The legacy writer accepts
Cartesian images with the corresponding VTK-aligned direction matrix
`[[0, 0, 1], [0, 1, 0], [1, 0, 0]]` and finite, positive spacing; it rejects
unrepresentable geometry before creating or truncating the output path.
These dimension and spacing requirements follow the
[VTK legacy dataset specification](https://docs.vtk.org/en/v9.6.1/vtk_file_formats/vtk_legacy_file_format.html#dataset-format).
The scalar image path accepts one component per point and rejects
multi-component declarations rather than discarding samples.

## Usage

```toml
[dependencies]
ritk-vtk = "0.2.0"
```
