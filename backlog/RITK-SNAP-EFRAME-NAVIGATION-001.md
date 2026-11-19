<a id="RITK-SNAP-EFRAME-NAVIGATION-001"></a>
## RITK-SNAP-EFRAME-NAVIGATION-001 — Paint pan and retain the press origin — todo
- milestone: M-viewer-reading
- outcome: the eframe image follows pan state and includes movement before the drag threshold.
- acceptance: real egui RawInput press/move/release changes textured-shape origin by the full raw displacement; rotated/anisotropic image overlays and picking stay aligned; clipping and comparison-pane behavior remain correct.
- scope: `crates/ritk-snap/src/app/{image_placement,viewport_render,tests}/`, primary/comparison viewport callers, manual
- next: translate painted and interacted image rectangles within the allocated clip and capture the actual press origin; test rendered shape bounds independently of production displacement helpers.
- basis: c4c271b5b253b0ec92880747dbc67933b96ef6ca
- needs: RITK-SNAP-NAVIGATION-UNITS-001, RITK-SNAP-IMAGE-FIT-001
- priority: correctness
