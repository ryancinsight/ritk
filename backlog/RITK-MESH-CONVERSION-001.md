<a id="RITK-MESH-CONVERSION-001"></a>

## RITK-MESH-CONVERSION-001 — Convert supported mesh formats — todo
- outcome: convert VTK/VTP, STL, OBJ, PLY, and GLB through RITK mesh models.
- acceptance: geometry, coordinate frame, normals, topology, and supported attributes round-trip; unsupported attributes return typed loss before output.
- scope: crates/ritk-vtk/, crates/ritk-io/, mesh tests and manual
- next: compare each mesh codec model and write a generic value-level conformance suite.
- basis: c58d8906bee373a7d2f87468fafc140310b9ec70
- status: todo
- needs: none
- priority: architecture
