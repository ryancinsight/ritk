<a id="RITK-SNAP-OBLIQUE-SESSION-MODULES-001"></a>

## RITK-SNAP-OBLIQUE-SESSION-MODULES-001 — Separate native oblique session concerns — blocked
- outcome: place plane rendering and gesture reduction in cohesive session modules.
- acceptance: extraction preserves rendered frames, retained-valid-frame behavior, and event outcomes.
- scope: `crates/ritk-snap/src/presentation/native_session/oblique.rs`
- next: separate render/rebuild state from the gesture reducer.
- basis: bf589f94b9826be8b4b05c85e5da31c338f01bab
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-OBLIQUE-APP-ADAPTER-001
- priority: architecture
