<a id="RITK-SNAP-NATIVE-ARCHIVE-001"></a>
## RITK-SNAP-NATIVE-ARCHIVE-001 — Keep a local study archive — blocked
- outcome: Imported studies persist in a local database with keywords, search and CSV export of the study list.
- acceptance: Tests import the public fixtures and assert query results, keyword edits and the exact CSV rows.
- scope: `crates/ritk-snap/src/` new archive module, `crates/ritk-snap/src/presentation/native_session/window_controls/`
- next: Specify the storage layout in an ADR first.
- basis: 30b8ba098
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-NATIVE-OPEN-001
- milestone: M-viewer-access
- priority: feature
