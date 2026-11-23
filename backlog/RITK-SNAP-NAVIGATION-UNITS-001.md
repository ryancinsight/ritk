<a id="RITK-SNAP-NAVIGATION-UNITS-001"></a>
## RITK-SNAP-NAVIGATION-UNITS-001 — Keep navigation in display coordinates — todo
- priority: correctness; milestone: M-viewer-reading
- outcome: pan, zoom, and window/level respond in display coordinates while annotation picking stays in source coordinates.
- acceptance: native render origin follows raw [24,-12] pixels at fits 0.5/1/2 and every orientation; zoom 1 with vertical 24 pixels becomes 0.76; window center/width 40/400 becomes 88/496 at sensitivity 4; batching, capture, browser backing-pixel policy, invalid input, and measurement ownership have independent value tests.
- [x] [patch] Firefox and Chromium pass the workstation journey. Exercise the public MRI and a 2×2 CT image at two window sizes and four rotations. Assert the final displacement even when the host coalesces every move before release. Exercise a 24-pixel drag through both host entry points, including rotations and a change of fit. The locked Métis already supplies press-relative displacement.
- scope: `crates/ritk-snap/src/{app,presentation,tools,ui}/`, ADRs 0028/0043, migration guide, manual; upstream `metis-core::input`
- basis: c4c271b5b253b0ec92880747dbc67933b96ef6ca
- needs: none
