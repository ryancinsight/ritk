<a id="RITK-SNAP-NATIVE-LOAD-001"></a>
## RITK-SNAP-NATIVE-LOAD-001 — Browse a study while it loads — todo
- outcome: Native study loading runs off the UI thread with progress and cancel, so images already read can be browsed.
- acceptance: A native test observes partial availability and cancel, and asserts the final pixels equal a synchronous load.
- scope: `crates/ritk-snap/src/presentation/native_session/{series,startup,events}.rs`, `crates/ritk-snap/src/app/load_tasks.rs`
- next: Move the load task out of the eframe gate.
- basis: 30b8ba098
- status: todo
- needs: none
- milestone: M-viewer-compare
- priority: tightening
