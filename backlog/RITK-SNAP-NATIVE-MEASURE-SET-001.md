<a id="RITK-SNAP-NATIVE-MEASURE-SET-001"></a>
## RITK-SNAP-NATIVE-MEASURE-SET-001 — Complete the measurement and annotation set — todo
- outcome: Cobb angle, polygon, open polygon, deviation distance, arrow, pencil, manual length calibration, and select/move/delete work natively.
- acceptance: Value tests assert each measurement on analytical geometry; deletion and move update the drawn result and persist through save/restore.
- scope: `crates/ritk-snap/src/tools/**`, `crates/ritk-snap/src/app/pointer_ops.rs`, `crates/ritk-snap/src/presentation/native_session/layout/`
- next: Extend `Annotation` and `ToolKind` one tool per increment.
- basis: 55a600e3fbd828bf0e3be48a067d974d206d6e1d
- needs: none
- milestone: M-viewer-reading
- priority: feature
