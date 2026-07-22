# VTK Format Boundary

The VTK boundary is broader than the other image-format chapters because ritk uses it for both images and geometric data. `ritk-io::format::vtk` re-exports readers and writers for legacy VTK image data, XML image data (`.vti`), polydata (`.vtk`/`.vtp`), structured grids, unstructured grids, and common mesh formats such as OBJ, PLY, STL, and glTF. For volume workflows, the native `VtkReader` and `VtkWriter` round-trip a `ritk-image::Image<f32, B, 3>` while preserving shape and physical metadata. For surface workflows, the boundary exposes domain types like `VtkPolyData` and helpers such as `write_mesh_as_vtk`.

That makes VTK a natural integration point between Atlas imaging and downstream geometry tools. Coeus-backed images can be exported for inspection or imported from external pipelines, while mesh and contour outputs from registration or segmentation can stay in VTK-native structures until the final handoff. The chapter therefore focuses on where the file format ends, where `ritk-image` begins, and how surface-oriented outputs coexist with the same Atlas stack that powers tensor, backend, and registration execution.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Dedicated VTK image round trip | Planned | Show legacy VTK image import/export through `VtkReader` and `VtkWriter`. |
| Mesh and polydata export walkthrough | Planned | Cover `write_mesh_as_vtk`, polydata writers, and geometry handoff from image-derived surfaces. |
