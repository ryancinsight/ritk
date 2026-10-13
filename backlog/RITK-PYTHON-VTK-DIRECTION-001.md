<a id="RITK-PYTHON-VTK-DIRECTION-001"></a>

## RITK-PYTHON-VTK-DIRECTION-001 — Python images cannot be written to a no-direction format (legacy VTK, Analyze 7.5) — todo
- outcome: a Python-created image reaches a legacy-VTK or Analyze 7.5 file with its geometry intact, or the restriction is a recorded, accepted product decision rather than a silent dead path. Both formats store no direction field, so an axis-permuted image is either re-expressed (a pure axis permutation, no resampling needed) or refused before a destination opens.
- acceptance: either `rio.write_image` handles the permutation direction `numpy_array_direction()` produces and a roundtrip test proves values + geometry survive, or the restriction is documented at the `write_image` contract with the typed errors asserted by `test_write_image_vtk_rejects_non_identity_direction` (legacy VTK) and `test_write_image_analyze_rejects_non_identity_direction` (Analyze 7.5).
- scope: `crates/ritk-python/src/image.rs` (constructor direction), `crates/ritk-vtk/src/io/writer.rs` and `crates/ritk-analyze/src/writer.rs` (representability checks), `crates/ritk-python/tests/test_coverage_gaps.py`, `crates/ritk-python/tests/test_sitk_real_data_parity.py`.
- next: decide whether the legacy VTK and Analyze writers gain permutation support or the Python surface gains a direction parameter; until then the rejection tests (not roundtrips) are the correct gates.
- risk: silent axis misorientation in medical images; the fail-closed rejections exist precisely to prevent it, so any writer change must preserve them for genuinely unrepresentable geometry.
- basis: 49dcdf05, f5c77f56c
- status: todo
- needs: none
- priority: correctness
