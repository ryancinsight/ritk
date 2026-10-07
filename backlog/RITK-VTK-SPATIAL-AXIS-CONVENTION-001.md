<a id="RITK-VTK-SPATIAL-AXIS-CONVENTION-001"></a>

## RITK-VTK-SPATIAL-AXIS-CONVENTION-001 — Align the VTK legacy reader/writer with the canonical RITK axis order — done
- outcome: VTK legacy structured-points `SPACING` is written and read in the axis order the same file's `DIMENSIONS` line names, so the two header fields agree and an external VTK reader sees the correct voxel pitch.
- acceptance: `write_vtk` emits `SPACING` reversed from RITK `[Δdepth, Δrow, Δcol]`; `read_vtk` reverses the file triple back; a hand-built file with `DIMENSIONS 4 3 2` and `SPACING 0.5 0.75 1.25` reads back as RITK spacing `[1.25, 0.75, 0.5]`; the emitted `SPACING` line names the same axis as `DIMENSIONS`.
- scope: crates/ritk-vtk/src/io/{mod.rs,reader.rs,writer.rs}
- next: Landed. `write_vtk` passed `[spacing[0], spacing[1], spacing[2]]` straight into `SPACING`, while `encode_vtk_flat` takes the X extent from `dims[2]` — so the emitted file named the *column* axis in `DIMENSIONS` and gave that same axis the *depth* pitch in `SPACING`. The file contradicted itself, and the reader mirrored the error, so the round-trip suite stayed green. The false premise was documented in two places: `reader.rs:10` ("RITK spatial metadata (`Point`, `Spacing`) also uses [X, Y, Z] order, so values transfer directly without permutation") and `writer.rs:12`. The reversal now lives in one place, `io::reverse_spatial_axes`, shared by both sides; `ORIGIN` is a scanner-space position and is deliberately not reversed. Two falsifiable oracles were added — the hand-built-file read oracle and the byte-level `SPACING`/`DIMENSIONS` agreement oracle. Verified by reverting the two call sites and re-running: both failed.
- basis: 733781c68ebe1ef6d9e09e128516ab9bf2f16ffe
- status: done
- needs: none
- priority: correctness
