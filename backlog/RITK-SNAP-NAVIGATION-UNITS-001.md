<a id="RITK-SNAP-NAVIGATION-UNITS-001"></a>
## RITK-SNAP-NAVIGATION-UNITS-001 — Keep navigation in display coordinates — done [patch]
- priority: correctness; milestone: M-viewer-reading
- outcome: pan, zoom, and window/level respond in display coordinates while annotation picking stays in source coordinates.
- acceptance: native render origin follows raw [24,-12] pixels at fits 0.5/1/2 and every orientation; zoom 1 with vertical 24 pixels becomes 0.76; window center/width 40/400 becomes 88/496 at sensitivity 4; batching, capture, browser backing-pixel policy, invalid input, and measurement ownership have independent value tests.
- [x] [patch] A 24-pixel drag produces the same rendered displacement at three native fits and four rotations on two image shapes. Firefox and Chromium verify pan and zoom at two window sizes on the public MRI and 2×2 CT. A coalesced drag uses the release position. [Windows tests](https://github.com/ryancinsight/ritk/actions/runs/45105768737/job/137467394703) · [browser matrix](https://github.com/ryancinsight/ritk/actions/runs/45105771232)
- scope: `crates/ritk-snap/src/{app,presentation,tools,ui}/`, ADRs 0028/0043, migration guide, manual; upstream `metis-core::input`
- basis: c4c271b5b253b0ec92880747dbc67933b96ef6ca
- needs: none
