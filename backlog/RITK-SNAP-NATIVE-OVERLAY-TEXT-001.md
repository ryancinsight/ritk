<a id="RITK-SNAP-NATIVE-OVERLAY-TEXT-001"></a>
## RITK-SNAP-NATIVE-OVERLAY-TEXT-001 — Draw clinical image overlays natively — todo
- outcome: Each image pane shows demographics corners, orientation letters, a scale ruler and image number, with a toggle and an anonymized mode. The pane reads out the value under the cursor.
- acceptance: Native tests assert letters for known direction cosines, ruler length against pixel spacing, and that anonymized mode renders no patient text.
- scope: `crates/ritk-snap/src/presentation/native_session/layout/{overlay,compare}.rs`, `crates/ritk-snap/src/ui/overlay`, `crates/ritk-snap/src/app/pointer_ops.rs`
- next: Port `ui/overlay` to the native layout. The spacing line, axis table and overflow disclosure are in `414da7491:crates/ritk-snap/src/ui/overlay/details.rs`.
- basis: 30b8ba098
- needs: none
- milestone: M-viewer-reading
- priority: feature
