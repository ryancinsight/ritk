<a id="RITK-SNAP-NATIVE-POINTER-CAPTURE-001"></a>
## RITK-SNAP-NATIVE-POINTER-CAPTURE-001 — End a drag released off the window — todo
- outcome: A button released outside the window ends its gesture at release, because the native host captures the pointer while a button is held.
- acceptance: A Windows native test presses inside the window, releases outside it, and asserts the session received the release and holds no pressed button.
- scope: `repos/moirai` window callback, `repos/metis` host events, `crates/ritk-snap/src/presentation/native_session/routing.rs`
- next: Add `SetCapture`/`ReleaseCapture` to the Moirai window callback, then advance Métis and RITK in dependency order.
- basis: 414da7491
- needs: none
- milestone: M-viewer-reading
- priority: correctness
