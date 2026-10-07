<a id="RITK-MESH-CONVERSION-001"></a>

## RITK-MESH-CONVERSION-001 — Convert supported mesh formats — todo
- outcome: convert VTK/VTP, STL, OBJ, PLY, and GLB through RITK mesh models.
- acceptance: geometry, coordinate frame, normals, topology, and supported attributes round-trip; unsupported attributes return typed loss before output.
- scope: crates/ritk-vtk/, crates/ritk-io/, mesh tests and manual
- next: compare each mesh codec model and write a generic value-level conformance suite.
- basis: 4ebc650d6e25a7a6775910ba23bc35c8c7cb78e4
- status: todo
- needs: none
- priority: architecture
