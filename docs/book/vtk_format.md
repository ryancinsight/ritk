# VTK Format Boundary

The VTK boundary supports scalar image handoff and surface-oriented output.
For scalar image paths, use the same native facade as the other lossless
formats:

~~~rust,ignore
let image = ritk_io::read_image_native("volume.vtk")?;
ritk_io::write_image_native("volume_copy.vtk", &image)?;
~~~

Surface output uses explicit mesh and polydata writer APIs exported by
ritk-io, including write_mesh_as_vtk and the OBJ, PLY, STL, VTP, and glTF
helpers. A surface writer does not infer image spacing from a raw point list;
construct the mesh in the intended physical frame before exporting it.

## Direction-aware volume handoff

`ritk-snap::LoadedVolume` uses `[depth, row, column]` axes and keeps its
direction cosine matrix beside the decoded samples. The VTK boundary converts
that contract to a zero-copy [`ritk_vtk::VtkImageVolume`] in VTK order
`[x, y, z] = [column, row, depth]`:

```rust,ignore
let volume = ritk_vtk::VtkImageVolume::try_from(&loaded_volume)?;
assert_eq!(volume.dimensions(), [columns, rows, depth]);
let samples = volume.scalars();
```

The carrier preserves origin, spacing, direction, channel count and
x-fastest interleaved samples while sharing the source allocation. Call
`to_vtk_image_data` only at an existing serializer or filter boundary that
requires VTK's owned `Vec<f32>` attribute arrays; that operation is explicit
and retains the direction matrix on the carrier.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Native VTK image boundary | Available | Uses the unified image facade for supported VTK image paths. |
| Mesh and polydata export boundary | Available | Uses explicit mesh and polydata writers for surface handoff. |
