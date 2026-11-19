<a id="RITK-SNAP-NATIVE-POINTER-CAPTURE-001"></a>
## RITK-SNAP-NATIVE-POINTER-CAPTURE-001 — End a drag released off the window — todo
- outcome: A real Windows host release outside the window reaches the session and ends the active gesture.
- acceptance: Drive a real host press inside and signed release outside the client area; assert the session received the release and holds no pressed button or gesture owner.
- scope: crates/ritk-snap/src/presentation/native.rs, crates/ritk-snap/src/presentation/native_session/tests/
- next: Verify the existing Moirai capture callback through Métis to RITK; injected WindowEvent tests cover the session but do not establish the host-to-session path.
- basis: 30ade461c3cbf50a286111a44213e195c2fe5d95
- needs: none
- milestone: M-viewer-reading
- priority: correctness
