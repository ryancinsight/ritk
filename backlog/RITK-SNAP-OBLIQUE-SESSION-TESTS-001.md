<a id="RITK-SNAP-OBLIQUE-SESSION-TESTS-001"></a>

## RITK-SNAP-OBLIQUE-SESSION-TESTS-001 — Verify oblique rendering and startup — blocked
- outcome: cover real session initialization, frame changes, resize, and retained-frame failures.
- acceptance: tests assert frame pixels and semantic state for valid and rejected transitions.
- scope: `crates/ritk-snap/src/presentation/native_session/tests/oblique/`
- next: add rendering and startup cases against manufactured volumes.
- basis: 3fcdc3dd
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-OBLIQUE-SESSION-WIRING-001
- priority: verification
