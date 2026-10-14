<a id="RITK-CASTFROM-MIGRATE"></a>

## RITK-CASTFROM-MIGRATE — Retire `CastFrom` from ritk — todo
- outcome: no ritk source uses eunomia `CastFrom`/`CastTo`; each site converts through std or a named eunomia method.
- acceptance: `git grep -c -E '\b(cast_from|cast_to|CastFrom|CastTo)\b' -- '*.rs'` is empty; ritk builds against the eunomia that drops `NumericElement: CastFrom<i32>`.
- scope: `crates/ritk-filter/examples/`, `crates/ritk-io/examples/`, `crates/ritk-registration/{examples,src/metric/mind}/`, `crates/ritk-interpolation/src/native.rs` (its `usize: CastFrom<T>` bounds).
- next: update the standalone lock to Eunomia main 8e231866, then map each float-to-integer site to its rounding contract.
- links: consumer of [EUNOMIA-CASTFROM-RETIRE](../../eunomia/backlog.md#EUNOMIA-CASTFROM-RETIRE); Eunomia PR #162 merged with `IntegerTarget::{from_rounded, try_from_rounded, from_floor}`.
- audit 2026-10-14 (static survey of the same tree; unverified by execution): no `CastFrom` in production voxel/dim/stride/offset/DICOM paths; remaining 10 sites are 8 display-byte conversions in examples (book_windowing_rescale.rs:70,158; book_processing_pipeline.rs:116; book_filter_gallery.rs:93; book_n4_bias.rs:82,109,111; book_demons_registration.rs:193; book_registration.rs:241 — three with silent NaN-to-0 saturation: book_n4_bias.rs:82, book_demons_registration.rs:193, book_registration.rs:241) plus 2 guarded index sites (mind/tests/geometry.rs:148 voxel index; book_windowing_rescale.rs:158 bin index). Migrate with `from_rounded` (note ties-to-even vs `round()` ties-away) or `try_from_rounded`/`from_floor`; `ritk-interpolation/src/native.rs` bound already gone.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: todo
- needs: none
- priority: architecture
