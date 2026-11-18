<a id="RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001"></a>
## RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001 — Verify complete native gestures — blocked
- milestone: M-viewer-compare
- outcome: test linked navigation, plane rotation, cine-independent event timing, and patient-length interaction.
- acceptance: end-to-end event sequences assert exact cursor, orientation, measurement, and repaint outcomes.
- scope: `crates/ritk-snap/src/presentation/native_session/tests/oblique/gestures.rs`, `measurement.rs`
- next: verify full pointer and keyboard workflows on rotated anisotropic data.
- basis: 3fcdc3dd
- needs: RITK-SNAP-OBLIQUE-ROUTING-001
- priority: verification
