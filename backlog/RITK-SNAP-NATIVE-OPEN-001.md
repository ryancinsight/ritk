<a id="RITK-SNAP-NATIVE-OPEN-001"></a>
## RITK-SNAP-NATIVE-OPEN-001 — Open files, archives and drops natively — todo
- outcome: Native users can open a single file, a DICOMDIR, an unencrypted ZIP archive, or dropped files, and reopen recent studies.
- acceptance: Native tests load each input kind and assert the same series pixels as the folder path; malformed archives return typed errors under a read budget.
- scope: `crates/ritk-snap/src/presentation/native_session/series.rs`, `crates/ritk-snap/src/launch.rs`, `crates/ritk-snap/src/dicom/input_path.rs`, metis host drop events
- next: Confirm the Métis file-dialog and drop surface, then route it.
- basis: 30b8ba098
- needs: none
- milestone: M-viewer-access
- priority: feature
