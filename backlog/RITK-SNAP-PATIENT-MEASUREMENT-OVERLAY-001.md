<a id="RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001"></a>

## RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001 — Present patient-space lengths — todo
- outcome: draw completed measurements, each on the plane it was placed on, with its value, over orthogonal panes; oblique panes reuse the mapping when they land.
- acceptance: label positions use the same plane projection as pixels; a 3–4–5 measurement displays 5 mm.
- scope: `crates/ritk-snap/src/presentation/native_session/layout/`, `crates/ritk-snap/src/tools/interaction/`, `crates/ritk-snap/src/session/`
- next: Record each annotation's plane, then project endpoints through the crosshair mapping. Label formats (`{:.1} mm`, `{:.1}°`, `μ={:.1} σ={:.1}`, `{:.0} HU`) and the bisector placement with its antiparallel fallback are in `414da7491:crates/ritk-snap/src/ui/measurements/mod.rs`.
- basis: 30b8ba098
- status: todo
- needs: none
- milestone: M-viewer-reading
- priority: correctness
