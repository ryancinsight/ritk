<a id="RITK-CASTFROM-MIGRATE"></a>

## RITK-CASTFROM-MIGRATE — Retire `CastFrom` while preserving rounding [patch] — in review
- outcome: no ritk source uses eunomia `CastFrom`/`CastTo`; each site converts through std or a named eunomia method.
- acceptance: `git grep -c -E '\b(cast_from|cast_to|CastFrom|CastTo)\b' -- '*.rs'` is empty; ritk builds against the eunomia that drops `NumericElement: CastFrom<i32>`.
- scope: `crates/ritk-filter/examples/`, `crates/ritk-io/examples/`, `crates/ritk-registration/{examples,src/metric/mind}/`, `crates/ritk-interpolation/src/native.rs` (its `usize: CastFrom<T>` bounds).
- next: review the production MIND half-tie correction tracked in RITK-MIND-HALF-TIE-001; the retirement and standalone Eunomia lock advance landed in PR831.
- links: consumer of [EUNOMIA-CASTFROM-RETIRE](../../eunomia/backlog.md#EUNOMIA-CASTFROM-RETIRE); Eunomia PR #162 merged with `IntegerTarget::{from_rounded, try_from_rounded, from_floor}`.
- executed follow-up 2026-10-14 [patch]: [PR831](https://github.com/ryancinsight/ritk/pull/831) lands the retirement at `8af30a9a8753c580bcb6aadc616e3a6bea25472f`, with Eunomia 0.9.1 `ee7c6911`. Its production MIND descriptor conversion changes half-away ties to even; three regression tests demonstrate changed descriptor levels and public similarity on Rust 1.97.0 / aarch64-apple-darwin. Track the contract-preserving correction and review status in [RITK-MIND-HALF-TIE-001](RITK-MIND-HALF-TIE-001.md). The older PR741 is still an open rescue draft and is preserved.
- audit 2026-10-14 (static survey of the same tree; unverified by execution): no `CastFrom` in production voxel/dim/stride/offset/DICOM paths; remaining 10 sites are 8 display-byte conversions in examples (book_windowing_rescale.rs:70,158; book_processing_pipeline.rs:116; book_filter_gallery.rs:93; book_n4_bias.rs:82,109,111; book_demons_registration.rs:193; book_registration.rs:241 — three with silent NaN-to-0 saturation: book_n4_bias.rs:82, book_demons_registration.rs:193, book_registration.rs:241) plus 2 guarded index sites (mind/tests/geometry.rs:148 voxel index; book_windowing_rescale.rs:158 bin index). Migrate with `from_rounded` (note ties-to-even vs `round()` ties-away) or `try_from_rounded`/`from_floor`; `ritk-interpolation/src/native.rs` bound already gone.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: in review; retirement landed, descriptor rounding follow-up remains.
- needs: none
- priority: correctness
