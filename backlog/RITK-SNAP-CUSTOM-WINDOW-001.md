<a id="RITK-SNAP-CUSTOM-WINDOW-001"></a>
## RITK-SNAP-CUSTOM-WINDOW-001 — Retain precise custom window settings — todo
- outcome: A numeric window dialog and named custom presets survive restart with add/apply/delete behavior.
- acceptance: Set exact center/width, add and delete presets, reload settings, and assert values, names, and rendered pixels; malformed settings preserve valid current state.
- scope: crates/ritk-snap/src/presentation/native_session/window_controls/, crates/ritk-snap/src/app/state.rs, repos/metis decimal entry
- next: Reuse existing decimal input and in-memory preset operations in a persisted settings model.
- basis: eb9a81dabb62777be2103fa16b8a2dcbab50f1e3
- needs: none
- milestone: M-viewer-reading
- priority: feature
- reference: RadiAnt 2021.2; [feature comparison](../docs/manual/features.md#reading-and-annotations)
