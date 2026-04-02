//! DICOM to NIfTI Converter Example
//!
//! This example loads a DICOM series and saves it as a NIfTI file.
//! If multiple series are found, it lists them all and loads the largest one.
//!
//! Usage:
//!   cargo run --example dicom_to_nifti -- <input_dicom_dir> <output_nifti_file>
//!
//! Example:
//!   cargo run --example dicom_to_nifti -- "D:\ritk\data\Paired MRI (T1, T2) and CT Scans Dataset\CT\DICOM\Patient_01" patient01_ct.nii.gz

use burn_ndarray::NdArray;
use ritk_io::{load_dicom_series, scan_dicom_directory, write_nifti};
use std::env;

type Backend = NdArray<f32>;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <input_dicom_dir> <output_nifti_file>", args[0]);
        eprintln!(
            "Example: {} \"D:\\ritk\\data\\Paired MRI (T1, T2) and CT Scans Dataset\\CT\\DICOM\\Patient_01\" patient01_ct.nii.gz",
            args[0]
        );
        std::process::exit(1);
    }

    let input_dir = &args[1];
    let output_file = &args[2];

    println!("Converting DICOM series to NIfTI...");
    println!("Input directory: {}", input_dir);
    println!("Output file: {}", output_file);

    let device = Default::default();

    // Scan for all DICOM series in the directory
    let series_list = scan_dicom_directory(input_dir)?;

    if series_list.is_empty() {
        anyhow::bail!("No DICOM series found in {}", input_dir);
    }

    println!("\nFound {} DICOM series:", series_list.len());
    for (i, series) in series_list.iter().enumerate() {
        println!(
            "  [{}] UID: {}  Modality: {}  Description: '{}'  Files: {}",
            i,
            series.series_instance_uid,
            series.modality,
            series.series_description,
            series.file_paths.len(),
        );
    }

    // Pick the series with the most files (typically the full volume)
    let best_index = series_list
        .iter()
        .enumerate()
        .max_by_key(|(_, s)| s.file_paths.len())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let selected = &series_list[best_index];
    println!(
        "\nSelected series [{}] with {} files",
        best_index,
        selected.file_paths.len()
    );

    // Load the selected series
    let image = load_dicom_series::<Backend>(selected, &device)?;
    println!("Loaded image with shape: {:?}", image.shape());
    println!("Spacing: {:?}", image.spacing());
    println!("Origin: {:?}", image.origin());

    // Write NIfTI
    write_nifti(output_file, &image)?;
    println!("Successfully saved NIfTI file: {}", output_file);

    Ok(())
}
