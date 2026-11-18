<a id="RITK-SNAP-ZOOM-SCALE-001"></a>
## RITK-SNAP-ZOOM-SCALE-001 — Correct the pixel-scale zoom preset — todo
- outcome: Zoom 1:1 and native image placement use the same spacing normalization.
- acceptance: A 200×100 panel with a 100×50 frame and square 0.5 mm spacing maps one texel to one display pixel; repeat for 1 mm and 2 mm spacing, anisotropic spacing, rotation, and degenerate inputs.
- scope: crates/ritk-snap/src/ui/zoom.rs, crates/ritk-snap/src/presentation/native_session/layout/, crates/ritk-snap/src/presentation/native_session/tests/display_controls.rs
- next: Derive the preset from the same normalized spacing as placement, then test observed native geometry rather than the helper alone.
- basis: 78a7f5936d5dd04165a723ad9d05ffaf669feb4e
- needs: none
- milestone: M-viewer-reading
- priority: correctness
- reference: [feature comparison](../docs/manual/features.md#reading-and-annotations); current helper gives 0.25 while placement produces 0.5 display pixels per texel for the first case.
