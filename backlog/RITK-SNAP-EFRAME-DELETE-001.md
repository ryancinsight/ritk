<a id="RITK-SNAP-EFRAME-DELETE-001"></a>
## RITK-SNAP-EFRAME-DELETE-001 — Delete the eframe entry points — todo
- outcome: Nothing starts the eframe shell: the CI `--features eframe-shell` example build (ci.yml:322), the `--eframe` captures in `scripts/viewer.py`, `examples/dicom_workflow.rs` with its `[[example]]` entry, the `run_eframe_*` re-exports (`src/lib.rs:47-50`) and functions, `launch/capture.rs`, the `--eframe` arm of `main.rs`, `impl eframe::App for EguiApp` and `update_orthogonal_surface` are gone. `EguiApp`, `EguiRenderState` and `RtDoseOverlayCacheEntry` stay in `app/eframe.rs` with `app/mod.rs:36,106`, because nine other gated `app/` files import them; the file goes with the last `impl EguiApp` file.
- acceptance: CI's required clippy (`cargo clippy --workspace --all-targets --all-features -- -D warnings`, ci.yml:169) builds `eframe-shell` and passes on every PR, dead-code warnings included, with no `dead_code` allow or expect added; a callee left unreferenced is deleted in the same PR, split by module. Default-build nextest passes and `git grep run_eframe` is empty. No local run uses the feature; the CI run is the evidence. Each PR stays within 400 changed lines.
- scope: `.github/workflows/ci.yml`, `scripts/viewer.py`, `crates/ritk-snap/{Cargo.toml,examples/dicom_workflow.rs,src/lib.rs,src/main.rs,src/launch.rs,src/launch/capture.rs,src/app/eframe.rs}`
- next: Delete the CI step, the example with its `[[example]]` entry, and the `viewer.py` captures first; they change no library code. Size the next PR from the dead-code report CI gives on that one.
- basis: 6d0f66501
- needs: none
- milestone: M-viewer-reading
- priority: architecture
