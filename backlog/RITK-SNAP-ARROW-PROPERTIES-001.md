<a id="RITK-SNAP-ARROW-PROPERTIES-001"></a>
## RITK-SNAP-ARROW-PROPERTIES-001 — Edit arrow text and appearance — todo
- priority: feature; milestone: M-viewer-reading
- outcome: arrow properties support comment text, style, size, color, opacity, named presets, and a default preset.
- acceptance: native creation/editing and restart tests assert exact text/style/size/color/opacity and preset identity; cancellation preserves the prior mark; rendered labels and arrow geometry agree.
- scope: `crates/ritk-snap/src/{app,presentation,tools,session}/`, reusable Métis property controls, manual
- next: use reference manual pages 98–101 to specify property and preset workflows; reuse mark selection and keep persisted annotation meaning in RITK.
- basis: c4c271b5b253b0ec92880747dbc67933b96ef6ca
- needs: RITK-SNAP-MARK-EDIT-001
