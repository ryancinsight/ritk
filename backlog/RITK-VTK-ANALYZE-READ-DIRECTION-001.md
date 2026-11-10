<a id="RITK-VTK-ANALYZE-READ-DIRECTION-001"></a>

## RITK-VTK-ANALYZE-READ-DIRECTION-001 — Read VTK and Analyze volumes with file axes — todo
- outcome: A legacy VTK or Analyze 7.5 volume reads with file X advancing world X. Today both readers return the `[z, y, x]` buffer with `Direction::identity()`, so under `p = origin + D·diag(spacing)·index` tensor axis 0 (file Z) advances world X.
- acceptance: `read_vtk` and `read_analyze` return the anti-diagonal direction (columns in tensor-axis order: axis 0 → Z, axis 2 → X), the same mapping `ritk-metaimage/src/spatial.rs` applies to a file-identity transform. A fixture `DIMENSIONS 2 3 4 / SPACING 1 10 100` asserts the world point of tensor index (0,0,1) is origin + (1,0,0) and of (3,0,0) is origin + (0,0,300), for VTK and for Analyze; it fails at origin/main.
- scope: a shared signed-permutation direction routine in crates/ritk-image; crates/ritk-vtk/src/io/{reader.rs,mod.rs}; crates/ritk-analyze/src/{reader.rs,stored_series.rs,tests.rs,stored_series/tests.rs}; crates/ritk-io/src/format/vtk/mod.rs; ritk-io tests and CLI/diffusion callers of `read_vtk`/`read_analyze`.
- next: add the routine to ritk-image, then convert the VTK reader and its callers as the first slice and Analyze as the second (each under 400 lines). Found by the ritk#860 probe at 9002b5c82 (`ritk-vtk/src/io/reader.rs:191`, `ritk-analyze/src/reader.rs:188`).
- basis: 8daf17c7a5c98016a6bf541e11efc5f64fb7d62c
- milestone: M-viewer-reading
- status: todo
- needs: none
- priority: correctness
