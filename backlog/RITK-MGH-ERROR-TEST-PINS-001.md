<a id="RITK-MGH-ERROR-TEST-PINS-001"></a>

## RITK-MGH-ERROR-TEST-PINS-001 — Pin the MGH error fields and limits that mutants survive — todo
- outcome: Every field of the `MghError` variants the readers return, and every limit they enforce, is checked by a test that fails when the value is wrong. The judges of ritk#890 and ritk#891 found six surviving mutants and three unchecked variants; the production code is correct, the tests do not hold it.
- acceptance: each mutant fails a named test in `cargo nextest run -p ritk-mgh`: `InvalidVersion.expected` built from the found value (`expected: version`); the header byte count ignoring frames (needs a multi-frame fixture); the `voxels` field of the byte-count error; the series MGZ cap (`take` limit); `read_mgh` passing `u64::MAX` as its limit; the SignedShort width changed to 4. `Open`, `Header` and `Image` variants are matched with field values outside doctests. The `# Errors` list exists once and the four readers link to it.
- scope: crates/ritk-mgh/src/reader/{mod.rs,tests/errors.rs}, crates/ritk-mgh/src/writer/tests/datatypes.rs, crates/ritk-mgh/src/error.rs
- next: write `InvalidVersion { expected: 1, found: 2 }` at `reader/tests/errors.rs` and `writer/tests/datatypes.rs`, then add the multi-frame fixture.
- basis: cd8a829ce11bc0adc097dc3b1f5cf950751ee5ec
- milestone: M-viewer-reading
- status: todo
- needs: none
- priority: verification
