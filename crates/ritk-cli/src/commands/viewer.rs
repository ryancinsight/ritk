//! DICOM viewer CLI command.
//!
//! This command bridges the CLI and `ritk-snap` viewer core. It loads a DICOM
//! study from disk, including flat folders and DICOMDIR-backed studies when the
//! underlying reader supports them, then prints a concise geometry summary.
//!
//! The command is intentionally headless. `ritk-snap` owns the viewer domain
//! model and backend abstraction; a future GUI backend can consume the same
//! core types without changing the import path or study-loading logic.
#![expect(
    clippy::print_stdout,
    reason = "RITK-LINT-1: ritk-cli is the application output layer"
)]

use anyhow::{bail, Context, Result};
use clap::Parser;
use ritk_io::{DicomReadMetadata, DicomSeriesInfo, DicomSliceMetadata, DicomStudyCatalog};
use ritk_snap::GeometrySummary;
use std::path::PathBuf;

use super::Backend;

/// Inspect a DICOM study from the command line.
#[derive(Debug, Clone, Parser)]
pub struct ViewerArgs {
    /// Path to a DICOM directory. The directory may contain a `DICOMDIR`
    /// file and referenced image files, or it may be a flat folder of DICOM
    /// Part 10 files.
    #[arg(value_name = "PATH")]
    pub path: PathBuf,

    /// Select this SeriesInstanceUID when the study contains multiple series.
    #[arg(long, value_name = "UID")]
    pub series_instance_uid: Option<String>,

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

/// Run the viewer command.
pub fn run(args: ViewerArgs) -> Result<()> {
    let path = args.path.as_path();

    let catalog = DicomStudyCatalog::scan(path)
        .with_context(|| format!("failed to scan DICOM study at {}", path.display()))?;
    let selected_uid = match args.series_instance_uid {
        Some(uid) => uid,
        None => match catalog.series() {
            [] => bail!("no DICOM series found at {}", path.display()),
            [series] => series.series_instance_uid().to_owned(),
            series => bail!(
                "multiple DICOM series found at {}; select one with --series-instance-uid (available: {})",
                path.display(),
                series
                    .iter()
                    .map(DicomSeriesInfo::series_instance_uid)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        },
    };
    let selected_series = catalog.select(&selected_uid)?;
    let backend = Backend::default();
    let (image, metadata) = catalog
        .load(&selected_uid, &backend)
        .with_context(|| format!("failed to load DICOM series {selected_uid}"))?;

    print_summary(path, selected_series, &image);

    if args.geometry || args.slices {
        print_geometry(&metadata);
    }

    if args.slices {
        print_slice_table(&metadata.slices);
    }

    if args.summary {
        return Ok(());
    }

    Ok(())
}

fn print_summary(
    path: &std::path::Path,
    series: &DicomSeriesInfo,
    image: &ritk_image::Image<f32, Backend, 3>,
) {
    let shape = image.shape();
    println!("DICOM study: {}", path.display());
    println!("  series_uid: {}", series.series_instance_uid());
    println!("  modality: {}", series.modality());
    println!("  description: {}", series.series_description);
    println!("  slices: {}", series.file_paths.len());
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
            series_instance_uid: None,
            geometry: true,
            slices: true,
            summary: false,
        };
        assert_eq!(args.path, PathBuf::from("test_data/2_skull_ct"));
        assert_eq!(args.series_instance_uid, None);
        assert!(args.geometry);
        assert!(args.slices);
        assert!(!args.summary);
    }

    #[test]
    fn viewer_args_parse_explicit_series_selection() {
        let args =
            ViewerArgs::try_parse_from(["viewer", "study", "--series-instance-uid", "2.25.73002"])
                .expect("viewer arguments");
        assert_eq!(args.path, PathBuf::from("study"));
        assert_eq!(args.series_instance_uid.as_deref(), Some("2.25.73002"));
    }

    #[test]
    fn test_print_slice_table_handles_empty_input() {
        let empty = Vec::<DicomSliceMetadata>::new();
        print_slice_table(&empty);
    }
}
