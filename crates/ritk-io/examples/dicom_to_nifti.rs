//! Analyze 7.5 to NIfTI Converter Example
//!
//! This example loads an Analyze 7.5 image pair and saves it as a NIfTI file.
//!
//! Usage:
//!   cargo run --example dicom_to_nifti -- <input_analyze_hdr_or_img> <output_nifti_file>
//!
//! Example:
//!   cargo run --example dicom_to_nifti -- "D:\ritk\data\brain.hdr" patient01_ct.nii.gz

use coeus_core::SequentialBackend;
use ritk_io::{
    format::{analyze::AnalyzeReader, nifti::native::NiftiWriter},
    ImageReader, ImageWriter,
};
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!(
            "Usage: {} <input_analyze_hdr_or_img> <output_nifti_file>",
            args[0]
        );
        eprintln!(
            "Example: {} \"D:\\ritk\\data\\brain.hdr\" patient01_ct.nii.gz",
            args[0]
        );
        std::process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];

    println!("Converting Analyze 7.5 image to NIfTI...");
    println!("Input file: {}", input_file);
    println!("Output file: {}", output_file);

    let image = AnalyzeReader::new(SequentialBackend).read(input_file)?;
    println!("Loaded image with shape: {:?}", image.shape());
    println!("Spacing: {:?}", image.spacing());
    println!("Origin: {:?}", image.origin());

    NiftiWriter::new(SequentialBackend).write(output_file, &image)?;
    println!("Successfully saved NIfTI file: {}", output_file);

    Ok(())
}
