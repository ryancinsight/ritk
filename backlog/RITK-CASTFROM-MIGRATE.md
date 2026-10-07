<a id="RITK-CASTFROM-MIGRATE"></a>

## RITK-CASTFROM-MIGRATE — Retire `CastFrom` from ritk — todo
- outcome: no ritk source uses eunomia `CastFrom`/`CastTo`; each site converts through std or a named eunomia method.
- acceptance: `git grep -c -E '\b(cast_from|cast_to|CastFrom|CastTo)\b' -- '*.rs'` is empty; ritk builds against the eunomia that drops `NumericElement: CastFrom<i32>`.
- scope: `crates/ritk-filter/examples/`, `crates/ritk-io/examples/`, `crates/ritk-registration/{examples,src/metric/mind}/`, `crates/ritk-interpolation/src/native.rs` (its `usize: CastFrom<T>` bounds).
- next: update the standalone lock to Eunomia main 8e231866, then map each float-to-integer site to its rounding contract.
- links: consumer of [EUNOMIA-CASTFROM-RETIRE](../../eunomia/backlog.md#EUNOMIA-CASTFROM-RETIRE); Eunomia PR #162 merged with `IntegerTarget::{from_rounded, try_from_rounded, from_floor}`.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: todo
- needs: none
- priority: architecture
