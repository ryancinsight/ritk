//! The `valid_range` attribute and the `image-min` / `image-max` datasets.

use crate::{
    attrs::extract_numeric_range,
    datatype::{stored_type, StoredType},
    payload::read_payload,
};
use anyhow::{bail, Context, Result};
use consus_core::Datatype;
use consus_hdf5::attribute::Hdf5Attribute;
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use ritk_codecs::sample::SampleBuffer;

const IMAGE_MIN_PATH: &str = "minc-2.0/image/0/image-min";
const IMAGE_MAX_PATH: &str = "minc-2.0/image/0/image-max";

fn optional_path(file: &Hdf5File<std::fs::File>, path: &str) -> Result<Option<u64>> {
    match file.open_path(path) {
        Ok(address) => Ok(Some(address)),
        Err(consus_core::Error::NotFound { .. }) => Ok(None),
        Err(error) => Err(anyhow::anyhow!(
            "Cannot inspect optional MINC2 dataset {path}: {error}"
        )),
    }
}

/// One scalar or per-slice floating-point range dataset, widened to `f64`
/// exactly, with its shape.
fn read_range_dataset(
    file: &Hdf5File<std::fs::File>,
    address: u64,
    path: &str,
    slice_count: usize,
) -> Result<(Vec<f64>, Vec<usize>)> {
    let dataset = file
        .dataset_at(address)
        .map_err(|error| anyhow::anyhow!("Cannot read {path} metadata: {error}"))?;
    if dataset.layout != StorageLayout::Contiguous {
        bail!(
            "MINC2 dataset {path} uses {:?} storage; only Contiguous is supported",
            dataset.layout
        );
    }
    if !matches!(&dataset.datatype, Datatype::Float { .. }) {
        bail!(
            "MINC2 dataset {path} must use a floating-point datatype, got {:?}",
            dataset.datatype
        );
    }
    let stored: StoredType = stored_type(&dataset.datatype)
        .with_context(|| format!("Decode MINC2 dataset {path} datatype"))?;
    let dims = dataset.shape.current_dims().to_vec();
    let count = match dims.as_slice() {
        [] => 1,
        [count] if *count == slice_count => *count,
        _ => {
            bail!("MINC2 dataset {path} must be scalar or have shape [{slice_count}], got {dims:?}")
        }
    };
    let data_address = dataset
        .data_address
        .with_context(|| format!("MINC2 dataset {path} has no contiguous data address"))?;
    let samples = read_payload(file, data_address, stored, count)
        .with_context(|| format!("Read MINC2 dataset {path}"))?;
    let values = match samples {
        SampleBuffer::F32(values) => values.into_iter().map(f64::from).collect(),
        SampleBuffer::F64(values) => values,
        other => bail!(
            "MINC2 dataset {path} decoded as {}, expected a floating-point type",
            other.sample_type()
        ),
    };
    Ok((values, dims))
}

/// The `image-min` and `image-max` values, scalar or one per slice, or the
/// MINC default real range `[0, 1]` when both datasets are absent.
pub(crate) fn read_image_ranges(
    file: &Hdf5File<std::fs::File>,
    slice_count: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let minimum_address = optional_path(file, IMAGE_MIN_PATH)?;
    let maximum_address = optional_path(file, IMAGE_MAX_PATH)?;
    match (minimum_address, maximum_address) {
        (None, None) => Ok((vec![0.0], vec![1.0])),
        (Some(_), None) => bail!("MINC2 image-min exists but image-max is missing"),
        (None, Some(_)) => bail!("MINC2 image-max exists but image-min is missing"),
        (Some(minimum_address), Some(maximum_address)) => {
            let (minima, minimum_shape) =
                read_range_dataset(file, minimum_address, IMAGE_MIN_PATH, slice_count)?;
            let (maxima, maximum_shape) =
                read_range_dataset(file, maximum_address, IMAGE_MAX_PATH, slice_count)?;
            if minimum_shape != maximum_shape {
                bail!(
                    "MINC2 image-min/image-max shape mismatch: {minimum_shape:?} versus {maximum_shape:?}"
                );
            }
            Ok((minima, maxima))
        }
    }
}

/// The image dataset's `valid_range` attribute, or `default` when absent.
pub(crate) fn read_valid_range(
    attributes: &[Hdf5Attribute],
    default: [f64; 2],
) -> Result<[f64; 2]> {
    let Some(attribute) = attributes
        .iter()
        .find(|attribute| attribute.name == "valid_range")
    else {
        return Ok(default);
    };
    let value = attribute
        .decode_value()
        .map_err(|error| anyhow::anyhow!("Cannot decode MINC2 valid_range: {error}"))?;
    extract_numeric_range(&value).context("Invalid MINC2 valid_range")
}
