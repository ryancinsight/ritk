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

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Native VTK image boundary | Available | Uses the unified image facade for supported VTK image paths. |
| Mesh and polydata export boundary | Available | Uses explicit mesh and polydata writers for surface handoff. |
