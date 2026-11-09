<a id="RITK-SNAP-NATIVE-PACS-001"></a>
## RITK-SNAP-NATIVE-PACS-001 — Query and receive PACS studies natively — todo
- outcome: A native dialog performs echo, find and retrieve against configured nodes and accepts studies pushed to a storage listener.
- acceptance: Integration tests run against a local in-process SCP and assert the retrieved pixels; failures surface typed errors.
- scope: `crates/ritk-snap/src/pacs/`, `crates/ritk-snap/src/presentation/native_session/window_controls/`, `crates/ritk-io/src/format/dicom/networking/`
- next: Add a native dialog over `pacs/worker.rs`. Field ranges, the load-button rule and result formats are in `414da7491:crates/ritk-snap/src/ui/pacs_panel/mod.rs` and `results.rs`.
- basis: 30b8ba098
- status: todo
- needs: none
- milestone: M-viewer-access
- priority: feature
