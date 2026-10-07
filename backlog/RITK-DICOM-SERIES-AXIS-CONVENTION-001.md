<a id="RITK-DICOM-SERIES-AXIS-CONVENTION-001"></a>

## RITK-DICOM-SERIES-AXIS-CONVENTION-001 — Align DICOM series reader/writer with the canonical RITK axis order — done
- outcome: Every DICOM series read path and the series writer agree with the canonical RITK tensor-axis convention (`[depth, row, col] = [z, y, x]`, `Spacing<3> = [Δdepth, Δrow, Δcol]`) and with each other.
- acceptance: `write_dicom_series`/`write_dicom_series_native` emit `ImageOrientationPatient = [col-axis, row-axis]` and advance `ImagePositionPatient` along the depth axis; `series/loader.rs` assembles `Direction` columns `[normal, row, col]`; the `series`-module and `reader`-module loaders return identical voxels, spacing, origin, and direction for the same series; a canonical-direction volume round-trips index-preservingly.
- scope: crates/ritk-io/src/format/dicom/writer/series.rs, crates/ritk-io/src/format/dicom/series/loader.rs, crates/ritk-io/src/format/dicom/reader/{types.rs,scan/geometry.rs}, DICOM writer/reader tests
- next: Landed. `writer/series.rs` emitted `IOP = [columns[0], columns[1]]` and advanced IPP along `columns[2]`, while `series/loader.rs` built `Direction::from_columns([IOP[0..3], IOP[3..6], normal])` — the ITK axis order, which contradicts `writer/metadata.rs`, `reader/scan/geometry.rs::assemble_direction`, and the documented NIfTI/NRRD/MetaImage convention. The two RITK DICOM read paths returned different geometry for the same files. Fixed by swapping tensor axes 0 and 2 in both places. Chirality contract documented: `det(direction) = −1` (every RITK codec's output) round-trips index-preservingly; `det = +1` (e.g. `Direction::identity()`) still emits correct geometry but its slices advance against the DICOM normal, so a normal-sorting reader returns the volume with the depth index reversed.
- basis: 4b2d8b031c4ecdeec609295f8dbdff26c2cbeefb
- status: done
- needs: none
- priority: correctness
