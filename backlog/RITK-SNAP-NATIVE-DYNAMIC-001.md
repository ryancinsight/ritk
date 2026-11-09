<a id="RITK-SNAP-NATIVE-DYNAMIC-001"></a>
## RITK-SNAP-NATIVE-DYNAMIC-001 — Add time-intensity curves and DSA — blocked
- outcome: Dynamic series plot time-intensity curves for an ROI, and XA series support subtraction with auto and manual pixel shift.
- acceptance: Tests assert curve values and subtraction pixels on synthetic dynamic series with a known shift.
- scope: `crates/ritk-snap/src/` new dynamic module, `crates/ritk-snap/src/presentation/native_session/`
- next: Specify the frame-time model from DICOM attributes.
- basis: 30b8ba098
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-NATIVE-MEASURE-SET-001
- milestone: M-viewer-advanced
- priority: feature
