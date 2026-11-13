<a id="RITK-SNAP-EFRAME-REMOVAL-001"></a>
## RITK-SNAP-EFRAME-REMOVAL-001 — Remove the eframe shell — blocked
- outcome: Métis is the viewer's only shell: the `eframe-shell` feature and the egui, eframe and rfd dependencies are gone, and viewer logic that sat behind those gates is reachable from the native session.
- acceptance: `cargo tree -p ritk-snap` lists no egui, eframe or rfd; `git grep eframe-shell` is empty; `cargo nextest list -p ritk-snap` before and after differ only by tests whose subject was an egui widget.
- scope: `Cargo.toml`, `crates/ritk-snap/{Cargo.toml,README.md,patches.py,src/app/mod.rs,src/app/eframe.rs}`, the remaining `cfg(feature = "eframe-shell")` sites, `README.md`, `docs/adr/0026-viewer-presentation-migration.md`, `docs/architecture.md`, `docs/migration_selected_dicom.md`, `docs/manual/`
- next: File each slice of the plan in the PR that filed this block as the board budget allows. Then remove the feature, the dependencies, the `cfg` sites and the eframe docs here, each PR within 400 changed lines.
- basis: 6d0f66501
- status: blocked
- blocker: 19 slices of the plan in the PR that filed this block are unfiled and precede this item: annotation CSV, PET SUV readout, histogram geometry, RT-dose RGBA, overlay labels, MPR cursor, view transform, tool keys, session snapshot, dialog filters, preset references, render ungate, app ungate, and the deletions of app menus, app panels, viewports, the filter panel, side panels and image painters. The four `needs` are only the filed frontier. Re-open when the last of the 19 lands.
- needs: RITK-SNAP-EFRAME-PORT-001, RITK-SNAP-EFRAME-PORT-002, RITK-SNAP-EFRAME-PORT-003, RITK-SNAP-EFRAME-DELETE-001
- milestone: M-viewer-reading
- priority: architecture
