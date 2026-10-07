<a id="RITK-PYTHON-VTK-DIRECTION-001"></a>

## RITK-PYTHON-VTK-DIRECTION-001 — Python images cannot be written to legacy VTK — todo
- outcome: a Python-created image reaches a VTK file with its geometry intact, or the restriction is a recorded, accepted product decision rather than a silent dead path.
- acceptance: either `rio.write_image` handles the permutation direction `numpy_array_direction()` produces (axis permutation, no resampling needed) and a roundtrip test proves values + geometry survive, or the restriction is documented at the `write_image` contract with the typed error asserted by `test_write_image_vtk_rejects_non_identity_direction`.
- scope: `crates/ritk-python/src/image.rs` (constructor direction), `crates/ritk-vtk/src/io/writer.rs` (representability check), `crates/ritk-python/tests/test_coverage_gaps.py`.
- next: decide whether the legacy writer gains permutation support or the Python surface gains a direction parameter; until then the rejection test (not a roundtrip) is the correct gate.
- risk: silent axis misorientation in medical images; the fail-closed rejection exists precisely to prevent it, so any writer change must preserve it for genuinely unrepresentable geometry.
- basis: f095910f
- status: todo
- needs: none
- priority: correctness
