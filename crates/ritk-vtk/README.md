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

The legacy structured-points writer stores Cartesian images on the file axes
the reader returns (`ritk_image::file_axes_direction()`). An image with any
other signed axis permutation, identity included, is relabeled onto them so
every voxel keeps its world point; an oblique direction or a non-Cartesian map
is rejected before the output path is created or truncated.

## Usage

```toml
[dependencies]
ritk-vtk = "0.2.0"
```
