<a id="RITK-FORMAT-SPATIAL-CONTRACT-COVERAGE-001"></a>

## RITK-FORMAT-SPATIAL-CONTRACT-COVERAGE-001 — Assert spatial metadata in the native writer/reader contract tests — done
- outcome: The shared native writer/reader contract test asserts spacing, origin, and direction parity, not only shape and voxels, and states the limit of what a round trip can prove.
- acceptance: `assert_native_writer_reader_round_trips` asserts spacing, origin, and direction for a non-isotropic spacing and a non-symmetric `det = −1` direction; every codec routed through it (NRRD, MetaImage, Analyze, MGH, MNC, TIFF, VTK) declares the fidelity it can carry, and codecs carrying no spatial metadata are named.
- scope: crates/ritk-io/src/format/tests_native_readers.rs
- next: Landed, with a correction. The helper now asserts shape, voxels, spacing, origin, and direction using `SPACING = [1.5, 0.75, 0.9]` (non-isotropic and non-monotonic) and a `det = −1`, non-symmetric direction, so neither a permutation nor a transpose cancels itself. It is parameterised by `SpatialFidelity` because TIFF carries no spatial metadata (`None`) and legacy VTK cannot carry a direction matrix (`SpacingAndOrigin`); VTK was added to the routed set. The correction: this closes the "codec drops or rewrites geometry asymmetrically" gap but *cannot* catch a self-consistent transposition, because a writer and reader that apply the same axis permutation round-trip exactly. That was established by falsification — with the pre-fix MGH implementation restored, `native_mgh_writer_reader_contract_round_trips` still passed. Axis-order correctness is therefore pinned by file-format oracles, not round trips: `ritk-mgh`'s hand-built-header and byte-layout tests, `ritk-mif`'s transform-less-header oracle, and `ritk-vtk`'s hand-built-file and `SPACING`/`DIMENSIONS` agreement oracles. The helper's doc comment now says exactly this and names those oracles.
- basis: 733781c68ebe1ef6d9e09e128516ab9bf2f16ffe
- status: done
- needs: none
- priority: correctness
