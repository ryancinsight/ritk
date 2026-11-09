<a id="RITK-SNAP-NATIVE-OVERLAY-TEXT-001"></a>
## RITK-SNAP-NATIVE-OVERLAY-TEXT-001 — Draw clinical image overlays natively — todo
- outcome: Each image pane shows demographics corners, orientation letters, a scale ruler and image number, with a toggle and an anonymized mode.
- acceptance: Native tests assert letters for known direction cosines, ruler length against pixel spacing, and that anonymized mode renders no patient text.
- scope: `crates/ritk-snap/src/presentation/native_session/layout/{overlay,compare}.rs`, `crates/ritk-snap/src/ui/overlay`
- next: Port the gated overlay model to the native layout.
- basis: 30b8ba098
- status: todo
- needs: none
- milestone: M-viewer-reading
- priority: feature
