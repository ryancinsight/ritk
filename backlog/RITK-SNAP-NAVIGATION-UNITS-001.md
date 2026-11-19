<a id="RITK-SNAP-NAVIGATION-UNITS-001"></a>
## RITK-SNAP-NAVIGATION-UNITS-001 — Keep navigation in display coordinates — todo
- priority: correctness; milestone: M-viewer-reading
- outcome: pan, zoom, and window/level respond in display coordinates while annotation picking stays in source coordinates.
- acceptance: native render origin follows raw [24,-12] pixels at fits 0.5/1/2 and every orientation; zoom 1 with vertical 24 pixels becomes 0.76; window center/width 40/400 becomes 88/496 at sensitivity 4; batching, capture, browser backing-pixel policy, invalid input, and measurement ownership have independent value tests.
- scope: `crates/ritk-snap/src/{app,presentation,tools,ui}/`, ADRs 0028/0043, migration guide, manual; upstream `metis-core::input`
- next: specify the intentional public-state/API change and browser sensitivity correction in ADRs; implement reusable press-relative displacement in Métis, then migrate all RITK navigation callers and delete inverse-image navigation paths.
- basis: c4c271b5b253b0ec92880747dbc67933b96ef6ca
- needs: none
