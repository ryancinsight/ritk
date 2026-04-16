//! Analyze 7.5 to NIfTI Converter Example
//!
//! This example loads an Analyze 7.5 image pair and saves it as a NIfTI file.
//!
//! Usage:
//!   cargo run --example dicom_to_nifti -- <input_analyze_hdr_or_img> <output_nifti_file>
//!
//! Example:
//!   cargo run --example dicom_to_nifti -- "D:\ritk\data\brain.hdr" patient01_ct.nii.gz

use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_io::{read_analyze, write_nifti};
use std::env;

type Backend = NdArray<f32>;

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

    let device = Default::default();

    let image: Image<Backend, 3> = read_analyze(input_file, &device)?;
    println!("Loaded image with shape: {:?}", image.shape());
    println!("Spacing: {:?}", image.spacing());
    println!("Origin: {:?}", image.origin());

    write_nifti(output_file, &image)?;
    println!("Successfully saved NIfTI file: {}", output_file);

    Ok(())
}
