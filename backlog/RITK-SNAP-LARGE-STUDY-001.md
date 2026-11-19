<a id="RITK-SNAP-LARGE-STUDY-001"></a>
## RITK-SNAP-LARGE-STUDY-001 — Measure native study resource bounds — todo
- outcome: Concurrent native studies have a measured memory bound and controlled decode/render baselines for their supported workload sizes.
- acceptance: Derive the live volume/frame/worker storage bound, measure peak and steady-state load/cancel/close cycles at two volume scales and the supported panel limit, and compare isolated timing/instruction baselines without changing workloads.
- scope: crates/ritk-snap/src/presentation/native_session/series_loads.rs, crates/ritk-snap/src/app/volume_input*, crates/ritk-snap/benches/
- next: Identify resident buffer ownership and record a bounded profiling workload before choosing resource or performance changes. Reference: RadiAnt 2021.2 page 9; [feature comparison](../docs/manual/features.md#reading-and-annotations)
- basis: eb9a81dabb62777be2103fa16b8a2dcbab50f1e3
- needs: none
- priority: verification; milestone: M-viewer-reading
