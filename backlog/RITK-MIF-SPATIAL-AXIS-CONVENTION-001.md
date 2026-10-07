<a id="RITK-MIF-SPATIAL-AXIS-CONVENTION-001"></a>

## RITK-MIF-SPATIAL-AXIS-CONVENTION-001 — Reverse `.mif` `vox:` sizes on the transform-less read path — done
- outcome: A `.mif` header without a `transform:` block yields RITK spacing `[Δdepth, Δrow, Δcol]`, matching the transform path and the writer's own `vox:` emission.
- acceptance: `read_mif` on a transform-less header with `vox: 0.5 0.75 1.25` returns `spacing == [1.25, 0.75, 0.5]`; the transform path is unchanged; shape stays `[depth, row, col]`.
- scope: crates/ritk-mif/src/reader.rs, crates/ritk-mif/src/reader_tests.rs
- next: Landed. The `vox:` field is X, Y, Z (MRtrix3) and `write_mif` emits it as `[Δcol, Δrow, Δdepth]`, but the no-transform branch did `Spacing::new([vox_sizes[0], vox_sizes[1], vox_sizes[2]])` — index-for-index — while `decompose_transform_affine`'s own fallbacks already treated `vox_sizes` as XYZ (`vox_sizes.get(2)` for `sz`, `.first()` for `sx`). The two read paths therefore disagreed, and no round trip could see it: `write_mif` always emits a `transform:`, so the branch is reached only by files this crate did not write. Fixed to `[vox_sizes[2], vox_sizes[1], vox_sizes[0]]`. A falsifiable oracle was added (`a_transform_less_header_reverses_vox_sizes_into_ritk_axis_order`); verified by reverting the single line and re-running, which failed with `spacing[0] is Δdepth and must be the file Z size 1.25, got 0.5`.
- basis: 733781c68ebe1ef6d9e09e128516ab9bf2f16ffe
- status: done
- needs: none
- priority: correctness
