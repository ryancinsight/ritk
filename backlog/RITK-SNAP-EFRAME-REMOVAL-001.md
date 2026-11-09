<a id="RITK-SNAP-EFRAME-REMOVAL-001"></a>
## RITK-SNAP-EFRAME-REMOVAL-001 — Remove the eframe shell — todo
- outcome: Métis is the viewer's only shell: the `eframe-shell` feature and the egui, eframe and rfd dependencies are gone, and viewer logic that sat behind those gates is reachable from the native session.
- acceptance: `cargo tree -p ritk-snap` lists no egui, eframe or rfd; `git grep eframe-shell` is empty; `cargo nextest list -p ritk-snap` before and after differ only by tests whose subject was an egui widget.
- scope: `Cargo.toml`, `crates/ritk-snap/`, `.github/workflows/ci.yml`, `scripts/viewer.py`, `README.md`, `docs/adr/0026-viewer-presentation-migration.md`, `docs/manual/`
- next: Give surviving logic native callers, then delete the gated shell.
- basis: 30b8ba098
- status: todo
- needs: none
- milestone: M-viewer-reading
- priority: architecture
