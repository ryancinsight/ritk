//! Copy one NIfTI volume while checking stored sample bits and image metadata.
//!
//! The reader and writer keep the encoded sample type and values separate from
//! intensity calibration. The round-trip check therefore compares the sample
//! bytes and every metadata field, rather than only checking that both calls
//! returned successfully.
//!
//! Run with `cargo run -p ritk-nifti --example nifti_stored_roundtrip -- scan.nii.gz copy.nii.gz`.
#![expect(clippy::print_stdout, reason = "ratchet RITK-LINT-1")]

use std::env;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use ritk_codecs::ByteOrder;
use ritk_image_io::ImageReadBudget;
use ritk_nifti::{read_nifti_stored, write_nifti_stored};

fn arguments() -> Result<(PathBuf, PathBuf)> {
    let mut arguments = env::args_os().skip(1);
    let input = arguments
        .next()
        .context("usage: nifti_stored_roundtrip <input.nii[.gz]> <output.nii[.gz]>")?;
    let output = arguments
        .next()
        .context("usage: nifti_stored_roundtrip <input.nii[.gz]> <output.nii[.gz]>")?;
    if arguments.next().is_some() {
        bail!("expected exactly an input and output path");
    }
    Ok((PathBuf::from(input), PathBuf::from(output)))
}

fn encoded_samples(volume: &ritk_image_io::StoredVolume) -> Result<Vec<u8>> {
    volume
        .samples()
        .encode(ByteOrder::LeastSignificantByteFirst)
        .context("failed to encode stored samples for comparison")
}

fn main() -> Result<()> {
    let (input_path, output_path) = arguments()?;
    let source = read_nifti_stored(&input_path, ImageReadBudget::DEFAULT)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    write_nifti_stored(&output_path, &source)
        .with_context(|| format!("failed to write {}", output_path.display()))?;
    let decoded = read_nifti_stored(&output_path, ImageReadBudget::DEFAULT)
        .with_context(|| format!("failed to read back {}", output_path.display()))?;

    if source.shape() != decoded.shape() {
        bail!(
            "shape changed from {:?} to {:?}",
            source.shape(),
            decoded.shape()
        );
    }
    if source.samples().sample_type() != decoded.samples().sample_type() {
        bail!(
            "stored sample type changed from {:?} to {:?}",
            source.samples().sample_type(),
            decoded.samples().sample_type()
        );
    }
    if encoded_samples(&source)? != encoded_samples(&decoded)? {
        bail!("stored sample bits changed across the NIfTI round trip");
    }
    if source.metadata() != decoded.metadata() {
        bail!("physical metadata changed across the NIfTI round trip");
    }
    if source.coordinate_map() != decoded.coordinate_map() {
        bail!("coordinate map changed across the NIfTI round trip");
    }
    if source.calibration() != decoded.calibration() {
        bail!("intensity calibration changed across the NIfTI round trip");
    }

    println!(
        "preserved {:?} {:?} volume from {} to {}",
        decoded.samples().sample_type(),
        decoded.shape(),
        input_path.display(),
        output_path.display()
    );
    Ok(())
}
