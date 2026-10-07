<a id="RITK-VTK-IMAGE-CONTRACT-001"></a>

## RITK-VTK-IMAGE-CONTRACT-001 — Separate VTK image semantics — todo
- outcome: Define the VTK image-data contract independently of mesh and scene formats.
- acceptance: Value cases distinguish scalar arrays, origin, spacing, direction, and layout from mesh and scene data for VTI and legacy image data.
- scope: crates/ritk-vtk/src/domain/vtk_data_object/volume.rs, VTK image I/O, tests, and manual
- next: Compare VTK image models and existing round trips against the stored-volume contract.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: todo
- needs: none
- priority: correctness
