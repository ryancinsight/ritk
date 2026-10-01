//! DICOM series to NIfTI converter example.
//!
//! This example selects one DICOM image series and writes it as NIfTI.
//!
//! Usage:
//!   cargo run --example dicom_to_nifti -- <input_dicom_series_dir> <output_nifti_file> [series_uid]
//!
//! Example:
//!   cargo run --example dicom_to_nifti -- "D:\ritk\data\dicom-study" patient01_ct.nii.gz
#![expect(clippy::print_stderr, reason = "ratchet RITK-LINT-1")]
#![expect(clippy::print_stdout, reason = "ratchet RITK-LINT-1")]

use coeus_core::SequentialBackend;
use ritk_io::{
    format::{dicom, nifti::native::NiftiWriter},
    ImageWriter,
};
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if !(3..=4).contains(&args.len()) {
        eprintln!(
            "Usage: {} <input_dicom_series_dir> <output_nifti_file> [series_uid]",
            args[0]
        );
        std::process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];
    let backend = SequentialBackend;

    let image = match args.get(3) {
        Some(series_uid) => {
            dicom::read_native_dicom_series_with_uid(input_file, series_uid, &backend)?
        }
        None => dicom::read_native_dicom_series(input_file, &backend)?,
    };

    println!("Converting DICOM series to NIfTI...");
    println!("Input file: {}", input_file);
    println!("Output file: {}", output_file);
    println!("Loaded image with shape: {:?}", image.shape());
    println!("Spacing: {:?}", image.spacing());
    println!("Origin: {:?}", image.origin());

    NiftiWriter::new(backend).write(output_file, &image)?;
    println!("Successfully saved NIfTI file: {}", output_file);

    Ok(())
}
