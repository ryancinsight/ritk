<a id="RITK-SNAP-OBLIQUE-SESSION-WIRING-001"></a>

## RITK-SNAP-OBLIQUE-SESSION-WIRING-001 — Wire the oblique session pane — blocked
- outcome: initialize and update the fourth pane from the active volume and patient cursor.
- acceptance: source replacement rebuilds the plane and presents current pixels; failure keeps the last valid frame and surfaces the error.
- scope: `crates/ritk-snap/src/presentation/native_session/{composition,session,events}.rs`
- next: connect the plane lifecycle to native session state.
- basis: 3fcdc3dd
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-OBLIQUE-SESSION-MODULES-001
- priority: correctness
