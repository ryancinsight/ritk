<a id="RITK-SNAP-NATIVE-FUSION-001"></a>
## RITK-SNAP-NATIVE-FUSION-001 — Fuse series and read SUV natively — blocked
- outcome: Series from different modalities overlay with adjustable opacity, and PET panes read SUVbw values. Compare panels swap primary and secondary, and the secondary series has its own window. A secondary series opens from another folder, and series loads target the primary or secondary slot.
- acceptance: Tests assert blended pixels for a known overlay and SUVbw against the published formula.
- scope: `crates/ritk-snap/src/render/fusion.rs`, `crates/ritk-snap/src/dicom/suv.rs`, `crates/ritk-snap/src/presentation/native_session/`
- next: Route `render_fused_slice` into a compare panel. Deleted shell source: `414da7491:crates/ritk-snap/src/app/viewport_compare.rs`, and the swap and opacity slider in `414da7491:crates/ritk-snap/src/app/toolbar.rs`.
- basis: 30b8ba098
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-NATIVE-LINK-001
- milestone: M-viewer-advanced
- priority: feature
