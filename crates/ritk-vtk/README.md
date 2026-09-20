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

## Usage

```toml
[dependencies]
ritk-vtk = "0.2.0"
```
