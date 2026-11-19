<a id="RITK-SNAP-ANNOTATION-TEXT-001"></a>
## RITK-SNAP-ANNOTATION-TEXT-001 — Position readable annotation labels — todo
- outcome: A reader can reposition measurement text and set its font size while labels retain the same measurement and slice identity.
- acceptance: Drag a selected label under rotated display geometry and assert its text rectangle and unchanged measured value; round-trip the chosen position and size.
- scope: crates/ritk-snap/src/tools/interaction/, crates/ritk-snap/src/presentation/native_session/layout/
- next: Define stored image-relative label placement and reuse existing glyph measurements. Reference: RadiAnt 2021.2; [feature comparison](../docs/manual/features.md#reading-and-annotations)
- basis: eb9a81dabb62777be2103fa16b8a2dcbab50f1e3
- needs: none
- priority: feature; milestone: M-viewer-reading
