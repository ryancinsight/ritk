<a id="RITK-MGH-SPATIAL-AXIS-CONVENTION-001"></a>

## RITK-MGH-SPATIAL-AXIS-CONVENTION-001 — Align MGH reader/writer with the canonical RITK axis order — done
- outcome: MGH read/write spatial metadata conforms to the canonical RITK tensor-axis convention (`[depth, row, col] = [z, y, x]`, `Spacing<3> = [Δdepth, Δrow, Δcol]`), matching NIfTI/NRRD/MetaImage, and a malformed header is a recoverable error rather than a panic.
- acceptance: `MghRasBlock::into_image_geometry` returns `Spacing<3> = [Δz, Δy, Δx]` and `Direction` columns `[z, y, x]` from the header's `(x, y, z)`-ordered `spacing` and `Mdc`; `ras_block_from_geometry` is its exact inverse, so `write_ras_block` emits the header in `(x, y, z)` order; non-isotropic spacing, a non-symmetric `det = −1` direction, and the origin all round-trip; zero, negative, or NaN header spacing is rejected.
- scope: crates/ritk-mgh/src/{spatial.rs,reader/mod.rs,writer/mod.rs,lib.rs}, crates/ritk-mgh/src/reader/tests/{geometry.rs,errors.rs}, crates/ritk-mgh/src/writer/tests/header.rs, docs/book/mgh_format.md, crates/ritk-io/src/format/tests_native_readers.rs
- next: Landed. The header orders `spacing` and `Mdc` by the x, y, z voxel axes (`docs/book/mgh_format.md`, `lib.rs`), but `write_mgh_flat` wrote `spacing[axis]` and `direction[(row, col)]` index-for-index, so RITK `Δdepth` landed in the header's x slot and RITK column 0 in `Mdc` column 0; the reader mirrored the transposition, making the pair self-consistent but wrong for every external MGH file. The reversal now lives only in `spatial.rs` behind `MghRasBlock`. Two falsifiable oracles were added: the hand-built-header `test_read_nondefault_spatial` and the byte-level `test_header_binary_layout`. Verified by running the new tests against the pre-fix implementation: all three failed (spacing `[0.5,0.75,1.25]` instead of `[1.25,0.75,0.5]`; header slot 0 `0.5` instead of `2.0`; a panic at `ritk-spatial/src/spacing.rs:181`). NRRD, MetaImage, Analyze, and MINC were audited and already honour the reversal.
- basis: 733781c68ebe1ef6d9e09e128516ab9bf2f16ffe
- status: done
- needs: none
- priority: correctness
