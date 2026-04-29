//! `ritk-snap` binary entry point.
//!
//! Launches the eframe/egui DICOM viewer application.
fn main() -> anyhow::Result<()> {
    ritk_snap::run_app()
}
