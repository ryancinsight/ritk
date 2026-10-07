<a id="RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001"></a>

## RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001 — Present patient-space lengths — blocked
- outcome: render persisted length endpoints and millimetre labels over the active plane.
- acceptance: label positions use the same plane projection as pixels; a 3–4–5 measurement displays 5 mm.
- scope: `crates/ritk-snap/src/presentation/native_session/layout/measurement.rs`, `crates/ritk-snap/src/ui/measurements/`
- next: project endpoints through the shared physical plane mapping.
- basis: ba37d37658c679ba354eccf904f8074a2a2aff5e
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-OBLIQUE-SESSION-WIRING-001
- priority: correctness
