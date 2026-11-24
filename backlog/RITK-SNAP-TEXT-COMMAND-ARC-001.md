<a id="RITK-SNAP-TEXT-COMMAND-ARC-001"></a>
## RITK-SNAP-TEXT-COMMAND-ARC-001 — Build display text without a second copy — todo
- outcome: `text_command` builds `DisplayCommand::DrawText` from `impl Into<Arc<str>>`, so a `&str` caller allocates once instead of twice and an `Arc<str>` caller shares its allocation.
- evidence: `crates/ritk-snap/src/presentation/native_session/layout/text.rs:34` takes `impl Into<String>` and `:40` copies it again through `Arc::from(text.into().as_str())`; `overlay.rs:442` passes a `&str` literal. A `String` caller costs one copy either way, so the gain is for `&str` and `Arc<str>` callers.
- acceptance: `cargo nextest run -p ritk-snap -E 'test(native_overlay_text_command_shares_arc)'` passes an `Arc<str>` through `text_command` and asserts `Arc::ptr_eq` with the `DrawText` text; it does not compile today because `Arc<str>` is not `Into<String>`.
- scope: crates/ritk-snap/src/presentation/native_session/layout/text.rs, crates/ritk-snap/src/presentation/native_session/layout/overlay.rs, crates/ritk-snap/src/presentation/native_session/layout/overlay_clinical.rs, crates/ritk-snap/src/presentation/native_session/layout/overlay_display.rs, crates/ritk-snap/src/presentation/native_session/tests/overlay_text.rs
- next: Change the parameter to `impl Into<Arc<str>>`, pass `&str` callers unchanged, and add the test.
- basis: fca0be6142ba212378f2dbc79b53bb678ea2d852
- needs: none
- priority: tightening; milestone: M-viewer-reading
