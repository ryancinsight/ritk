<a id="RITK-SNAP-EFRAME-REMOVAL-001"></a>
## RITK-SNAP-EFRAME-REMOVAL-001 — Remove the eframe shell — todo
- outcome: Métis is the viewer's only shell: the `eframe-shell` feature and the egui, eframe and rfd dependencies are gone, and viewer logic that sat behind those gates is reachable from the native session.
- acceptance: `cargo tree -p ritk-snap` lists no egui, eframe or rfd; `git grep eframe-shell` is empty; `cargo nextest list -p ritk-snap` before and after differ only by tests whose subject was an egui widget.
- scope: `Cargo.toml`, `crates/ritk-snap/{Cargo.toml,README.md,patches.py,rewire.py}`, the remaining `cfg(feature = "eframe-shell")` sites, `README.md`, `docs/adr/0026-viewer-presentation-migration.md`, `docs/architecture.md`, `docs/migration_selected_dicom.md`, `docs/manual/`
- next: File the next slices as these land, per the slice plan in the PR that filed this block: port the remaining egui-embedded logic (annotation CSV, PET SUV readout, histogram geometry, RT-dose RGBA, overlay labels, MPR cursor, view transforms, tool keys, session snapshot, dialog filters, preset references, render and app operations), delete painters callers first, then remove the feature, dependencies, `cfg` sites and eframe docs here; each PR stays within 400 changed lines.
- basis: 6d0f66501
- needs: RITK-SNAP-EFRAME-PORT-001, RITK-SNAP-EFRAME-PORT-002, RITK-SNAP-EFRAME-PORT-003, RITK-SNAP-EFRAME-DELETE-001
- milestone: M-viewer-reading
- priority: architecture
