<a id="RITK-PYTHON-VTK-DIRECTION-001"></a>

## RITK-PYTHON-VTK-DIRECTION-001 — Write permuted images to VTK and Analyze — todo
- outcome: Any image whose direction is a signed axis permutation reaches a legacy VTK or Analyze 7.5 file with its geometry intact; a non-permutation direction is still refused before a destination opens.
- acceptance: the writers treat the anti-diagonal direction (the file-faithful mapping of a `[z, y, x]` buffer, `ritk_image::file_axes_direction()`) as canonical, and re-express any other signed permutation to it by axis transpose, flip and origin move, including an identity-tagged image. Tests: (1) all 48 signed 3-D permutations round-trip values and world positions through each writer and reader; (2) a Python image written and read back keeps `to_numpy()` in `[Z, Y, X]` and its SimpleITK direction; (3) the existing non-permutation rejection tests stay.
- scope: crates/ritk-vtk/src/io/writer.rs, crates/ritk-analyze/src/writer.rs, the ritk-image permutation routine, crates/ritk-python/tests/test_coverage_gaps.py, crates/ritk-python/tests/test_sitk_real_data_parity.py.
- next: re-express in each writer through the shared routine; both readers return `file_axes_direction()` (ritk#863, ritk#869), and both writers refuse every other direction, identity included.
- risk: re-expressing to identity instead would write X and Z swapped for SimpleITK, VTK and ParaView, while in-memory round trips still pass because the readers undo the same swap.
- basis: 8daf17c7a5c98016a6bf541e11efc5f64fb7d62c
- milestone: M-viewer-reading
- status: todo
- needs: none
- priority: correctness
