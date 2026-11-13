<a id="RITK-SNAP-EFRAME-DELETE-001"></a>
## RITK-SNAP-EFRAME-DELETE-001 — Delete the eframe entry points — todo
- outcome: No binary, example, script or CI job starts the eframe shell: the CI `--features eframe-shell` example build, the eframe capture in `scripts/viewer.py`, `examples/dicom_workflow.rs`, the `run_eframe_*` functions and `launch/{capture,viewport}.rs`, the eframe arm of `main.rs` and `app/eframe.rs` are gone. Callers go before callees, so the remaining gated code still compiles under the feature until its own deletion.
- acceptance: Default-build clippy and nextest pass; `git grep run_eframe` is empty. Each PR stays within 400 changed lines; split by module.
- scope: `.github/workflows/ci.yml`, `scripts/viewer.py`, `crates/ritk-snap/{examples,src/launch*,src/main.rs,src/app/eframe.rs}`
- next: Delete the CI step and `scripts/viewer.py` capture first.
- basis: 6d0f66501
- needs: none
- milestone: M-viewer-reading
- priority: architecture
