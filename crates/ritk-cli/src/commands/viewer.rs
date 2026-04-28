//! DICOM viewer CLI command.
//!
//! This command bridges the CLI and `ritk-snap` viewer core. It loads a DICOM
//! study from disk, including flat folders and DICOMDIR-backed studies when the
//! underlying reader supports them, then prints a concise geometry summary.
//!
//! The command is intentionally headless. `ritk-snap` owns the viewer domain
//! model and backend abstraction; a future GUI backend can consume the same
//! core types without changing the import path or study-loading logic.

use anyhow::{Context, Result};
use burn::tensor::backend::Backend as BurnBackend;
use burn_ndarray::NdArray;
use clap::Parser;
use ritk_core::image::Image;
use ritk_io::{
    load_dicom_series, scan_dicom_directory, DicomReadMetadata, DicomSeriesInfo, DicomSliceMetadata,
};
use ritk_snap::{GeometrySummary, Study, ViewerBackend, ViewerCore, ViewerEvent, ViewerState};
use std::path::PathBuf;

/// CPU backend used by the CLI viewer.
type Backend = NdArray<f32>;

/// Inspect a DICOM study from the command line.
#[derive(Debug, Clone, Parser)]
pub struct ViewerArgs {
    /// Path to a DICOM directory. The directory may contain a `DICOMDIR`
    /// file and referenced image files, or it may be a flat folder of DICOM
    /// Part 10 files.
    #[arg(value_name = "PATH")]
    pub path: PathBuf,

    /// Print the geometry summary.
    #[arg(long)]
    pub geometry: bool,

    /// Print per-slice metadata.
    #[arg(long)]
    pub slices: bool,

    /// Print only the summary and exit.
    #[arg(long)]
    pub summary: bool,
}

/// Headless backend for CLI inspection.
#[derive(Debug, Default)]
struct HeadlessViewerBackend {
    state: ViewerState,
    last_event: Option<ViewerEvent>,
}

impl ViewerBackend for HeadlessViewerBackend {
    type Error = std::io::Error;

    fn initialize(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn load_study<B: burn::tensor::backend::Backend, const D: usize>(
        &mut self,
        _study: &Study<B, D>,
        state: &ViewerState,
    ) -> Result<(), Self::Error> {
        self.state = *state;
        Ok(())
    }

    fn update_state(&mut self, state: &ViewerState) -> Result<(), Self::Error> {
        self.state = *state;
        Ok(())
    }

    fn render(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn handle_event(&mut self, event: ViewerEvent) -> Result<(), Self::Error> {
        self.last_event = Some(event);
        Ok(())
    }
}

/// Run the viewer command.
pub fn run(args: ViewerArgs) -> Result<()> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let path = args.path.as_path();

    let series = scan_dicom_directory(path)
        .with_context(|| format!("failed to scan DICOM study at {}", path.display()))?;
    let image = load_dicom_series::<Backend, _>(path, &device)
        .with_context(|| format!("failed to load DICOM study at {}", path.display()))?;

    let study = Study::new(image.clone())
        .with_dicom(series.metadata.clone())
        .with_source(path.to_path_buf());

    let mut core = ViewerCore::<Backend, 3>::new();
    let loaded_event = core.load_study(study);
    let mut backend = HeadlessViewerBackend::default();
    backend.initialize()?;
    backend.load_study(core.study().expect("study must exist"), core.state())?;
    backend.handle_event(loaded_event)?;
    backend.render()?;

    print_summary(path, &series, &image);

    if args.geometry || args.slices {
        print_geometry(&series.metadata);
    }

    if args.slices {
        print_slice_table(&series.metadata.slices);
    }

    if args.summary {
        return Ok(());
    }

    Ok(())
}

fn print_summary(path: &std::path::Path, series: &DicomSeriesInfo, image: &Image<Backend, 3>) {
    let shape = image.shape();
    println!("DICOM study: {}", path.display());
    println!("  slices: {}", series.num_slices);
    println!("  shape:  [{}, {}, {}]", shape[0], shape[1], shape[2]);
    println!(
        "  origin: [{:.6}, {:.6}, {:.6}]",
        image.origin()[0],
        image.origin()[1],
        image.origin()[2]
    );
    println!(
        "  spacing: [{:.6}, {:.6}, {:.6}]",
        image.spacing()[0],
        image.spacing()[1],
        image.spacing()[2]
    );
}

fn print_geometry(meta: &DicomReadMetadata) {
    let summary = GeometrySummary::from_dicom(meta);
    println!("geometry:");
    println!(
        "  dimensions: [{}, {}, {}]",
        summary.dimensions[0], summary.dimensions[1], summary.dimensions[2]
    );
    println!(
        "  spacing: [{:.6}, {:.6}, {:.6}]",
        summary.spacing[0], summary.spacing[1], summary.spacing[2]
    );
    println!(
        "  origin: [{:.6}, {:.6}, {:.6}]",
        summary.origin[0], summary.origin[1], summary.origin[2]
    );
    println!(
        "  direction:\n    [{:.6}, {:.6}, {:.6}]\n    [{:.6}, {:.6}, {:.6}]\n    [{:.6}, {:.6}, {:.6}]",
        summary.direction[0],
        summary.direction[1],
        summary.direction[2],
        summary.direction[3],
        summary.direction[4],
        summary.direction[5],
        summary.direction[6],
        summary.direction[7],
        summary.direction[8]
    );
    println!("  modality: {:?}", meta.modality);
    println!("  series_uid: {:?}", meta.series_instance_uid);
    println!("  study_uid: {:?}", meta.study_instance_uid);
    println!(
        "  frame_of_reference_uid: {:?}",
        meta.frame_of_reference_uid
    );
}

fn print_slice_table(slices: &[DicomSliceMetadata]) {
    println!("slices:");
    for (idx, slice) in slices.iter().enumerate() {
        println!("  slice {idx}:");
        println!("    path: {}", slice.path.display());
        println!("    instance_number: {:?}", slice.instance_number);
        println!("    sop_instance_uid: {:?}", slice.sop_instance_uid);
        println!("    slice_location: {:?}", slice.slice_location);
        println!("    pixel_spacing: {:?}", slice.pixel_spacing);
        println!("    slice_thickness: {:?}", slice.slice_thickness);
        println!(
            "    image_position_patient: {:?}",
            slice.image_position_patient
        );
        println!(
            "    image_orientation_patient: {:?}",
            slice.image_orientation_patient
        );
        println!("    rescale_slope: {:.6}", slice.rescale_slope);
        println!("    rescale_intercept: {:.6}", slice.rescale_intercept);
        println!("    sop_class_uid: {:?}", slice.sop_class_uid);
        println!("    transfer_syntax_uid: {:?}", slice.transfer_syntax_uid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewer_args_path_round_trip() {
        let args = ViewerArgs {
            path: PathBuf::from("test_data/2_skull_ct"),
            geometry: true,
            slices: true,
            summary: false,
        };
        assert_eq!(args.path, PathBuf::from("test_data/2_skull_ct"));
        assert!(args.geometry);
        assert!(args.slices);
        assert!(!args.summary);
    }

    #[test]
    fn test_print_slice_table_handles_empty_input() {
        let empty = Vec::<DicomSliceMetadata>::new();
        print_slice_table(&empty);
    }

    #[test]
    fn test_viewer_core_event_flow() {
        let state = ViewerState::default();
        let event = ViewerEvent::Status {
            message: "ready".to_string(),
        };
        let mut backend = HeadlessViewerBackend::default();
        backend.initialize().expect("initialize");
        backend.update_state(&state).expect("update_state");
        backend.handle_event(event.clone()).expect("handle_event");
        backend.render().expect("render");
        assert_eq!(backend.state, state);
        assert_eq!(backend.last_event, Some(event));
    }
}
