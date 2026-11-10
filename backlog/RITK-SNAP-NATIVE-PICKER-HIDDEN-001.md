<a id="RITK-SNAP-NATIVE-PICKER-HIDDEN-001"></a>
## RITK-SNAP-NATIVE-PICKER-HIDDEN-001 — Stop the series picker confirming hidden rows — todo
- outcome: With a filter that matches nothing, the multiple-series picker stays open and loads nothing, from the keyboard and from its Open button.
- acceptance: Native tests select a row, type a filter with no matches, then assert that Down then Enter and a click on Open both leave the picker open with no series assigned.
- scope: `crates/ritk-snap/src/presentation/native_session/window_controls/multi_series.rs`, `crates/ritk-snap/src/presentation/native_session/window_controls/multi_series/tests.rs`
- next: `rebuild_matches` keeps `selected` when the filter hides it, and Enter in the list and the Open button then confirm it. Confirm only rows present in `matches`.
- basis: 13d8176bc
- status: todo
- needs: none
- milestone: M-viewer-reading
- priority: correctness
