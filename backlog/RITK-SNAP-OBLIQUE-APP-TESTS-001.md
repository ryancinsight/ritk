<a id="RITK-SNAP-OBLIQUE-APP-TESTS-001"></a>
## RITK-SNAP-OBLIQUE-APP-TESTS-001 — Verify oblique adapter boundaries — blocked
- milestone: M-viewer-compare
- outcome: test source changes, invalid pointers, and orientation limits through the app contract.
- acceptance: adversarial and boundary inputs return the specified actions and never mutate unrelated planes.
- scope: `crates/ritk-snap/src/app/tests/action_adapter/oblique.rs`
- next: add rotated, edge, and invalid-input cases.
- basis: 3fcdc3dd
- needs: RITK-SNAP-OBLIQUE-APP-ADAPTER-001
- priority: verification
