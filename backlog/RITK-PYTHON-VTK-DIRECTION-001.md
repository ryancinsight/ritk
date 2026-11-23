<a id="RITK-PYTHON-VTK-DIRECTION-001"></a>
## RITK-PYTHON-VTK-DIRECTION-001 — Write permuted images to VTK and Analyze — done [patch]
- outcome: Any image whose direction is a signed axis permutation reaches a legacy VTK or Analyze 7.5 file with its geometry intact; a non-permutation direction is still refused before a destination opens.
- acceptance: the writers treat the anti-diagonal direction (the file-faithful mapping of a `[z, y, x]` buffer, `ritk_image::file_axes_direction()`) as canonical, and re-express any other signed permutation to it by axis transpose, flip and origin move, including an identity-tagged image. Tests: (1) all 48 signed 3-D permutations round-trip values and world positions through each writer and reader; (2) a Python image written and read back keeps `to_numpy()` in `[Z, Y, X]` and its SimpleITK direction; (3) the existing non-permutation rejection tests stay.
- scope: crates/ritk-vtk/src/io/writer.rs, crates/ritk-analyze/src/writer.rs, the ritk-image permutation routine, crates/ritk-python/tests/test_coverage_gaps.py, crates/ritk-python/tests/test_sitk_real_data_parity.py.
- evidence: The [installed-wheel SimpleITK test](../crates/ritk-python/tests/test_sitk_real_data_parity.py) checks the source direction in Python, all 24 physical point/sample pairs after VTK and Analyze export, and an independent VTK reread. The Analyze assertion uses the RITK originator convention and a grid representable by its integer fields. Both Rust format suites exercise all 48 signed permutations and reject oblique directions.
- risk: re-expressing to identity instead would write X and Z swapped for SimpleITK, VTK and ParaView, while in-memory round trips still pass because the readers undo the same swap.
- basis: 729bbd78254c1eada68211225f350ecfebaa3666
- priority: correctness; milestone: M-viewer-reading
- needs: none
