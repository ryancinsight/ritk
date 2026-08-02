//! MINC2 reader: HDF5-based 3-D volumetric image import.
//!
//! # Algorithm
//!
//! 1. Open the file as HDF5 via `consus_hdf5::file::Hdf5File`.
//! 2. Navigate to `/minc-2.0/dimensions/` and read spatial dimension
//!    metadata (`start`, `step`, `length`, `direction_cosines`) from
//!    each of `xspace`, `yspace`, `zspace`.
//! 3. Navigate to `/minc-2.0/image/0/image` and read the dataset
//!    metadata (shape, datatype, storage layout).
//! 4. Parse the `dimorder` attribute to determine axis mapping.
//! 5. Validate integer `valid_range` and scalar/per-slice image ranges.
//! 6. Stream raw voxel chunks into the final `f32` buffer, applying integer
//!    real-value scaling without retaining a second volume-sized byte buffer.
//! 7. Construct `Image<f32, B, 3>` with spatial metadata derived from
//!    dimension attributes and the dimorder axis mapping.
//!
//! # Contiguous Storage Requirement
//!
//! The current implementation reads contiguously-stored datasets only.
//! Chunked datasets require B-tree traversal and per-chunk decompression
//! which will be added in a follow-up sprint.
//!
//! Integer scaling follows the MINC pixel-conversion specification:
//! <https://www.bic.mni.mcgill.ca/software/minc/prog_guide/node19.html>.
//! Scalar and first-spatial-axis image ranges follow the standard variable
//! definitions:
//! <https://www.bic.mni.mcgill.ca/software/minc/minc1_format/node5.html>.

use crate::{
    attrs::extract_numeric_range,
    convert::{decode_float_bytes, decode_raw_bytes_into},
    scaling::{default_integer_valid_range, IntegerScaling},
    spatial::{
        build_spatial_metadata, order_dimensions_by_dimorder, read_dimension_metadata,
        read_dimorder,
    },
    IMAGE_PATH,
};
use anyhow::{bail, Context, Result};
use consus_core::Datatype;
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use std::path::Path;

const IMAGE_MIN_PATH: &str = "minc-2.0/image/0/image-min";
const IMAGE_MAX_PATH: &str = "minc-2.0/image/0/image-max";
const VOXEL_READ_BYTES: usize = 8 * 1_024;

fn checked_product(values: &[usize], label: &str) -> Result<usize> {
    values.iter().copied().try_fold(1_usize, |product, value| {
        product
            .checked_mul(value)
            .with_context(|| format!("{label} element count overflows usize"))
    })
}

fn optional_path(file: &Hdf5File<std::fs::File>, path: &str) -> Result<Option<u64>> {
    match file.open_path(path) {
        Ok(address) => Ok(Some(address)),
        Err(consus_core::Error::NotFound { .. }) => Ok(None),
        Err(error) => Err(anyhow::anyhow!(
            "Cannot inspect optional MINC2 dataset {path}: {error}"
        )),
    }
}

fn read_image_range_dataset(
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
    let dims = dataset.shape.current_dims().to_vec();
    let count = match dims.as_slice() {
        [] => 1,
        [count] if *count == slice_count => *count,
        _ => {
            bail!("MINC2 dataset {path} must be scalar or have shape [{slice_count}], got {dims:?}")
        }
    };
    let element_size = dataset
        .datatype
        .element_size()
        .context("Variable-length MINC2 image-range datatype is unsupported")?;
    let total_bytes = count
        .checked_mul(element_size)
        .with_context(|| format!("MINC2 dataset {path} byte count overflows usize"))?;
    let data_address = dataset
        .data_address
        .with_context(|| format!("MINC2 dataset {path} has no contiguous data address"))?;
    let raw = ritk_core::io_bounds::read_bounded_with(total_bytes, |offset, destination| {
        file.read_contiguous_dataset_bytes(data_address, offset, destination)
    })
    .map_err(|error| anyhow::anyhow!("Failed to read MINC2 dataset {path}: {error}"))?;
    let values = decode_float_bytes(&raw, &dataset.datatype)
        .with_context(|| format!("Decode MINC2 dataset {path}"))?;
    if values.len() != count {
        bail!(
            "MINC2 dataset {path} decoded {} values, expected {count}",
            values.len()
        );
    }
    Ok((values, dims))
}

fn read_image_ranges(
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
                read_image_range_dataset(file, minimum_address, IMAGE_MIN_PATH, slice_count)?;
            let (maxima, maximum_shape) =
                read_image_range_dataset(file, maximum_address, IMAGE_MAX_PATH, slice_count)?;
            if minimum_shape != maximum_shape {
                bail!(
                    "MINC2 image-min/image-max shape mismatch: {minimum_shape:?} versus {maximum_shape:?}"
                );
            }
            Ok((minima, maxima))
        }
    }
}

fn read_valid_range(
    attributes: &[consus_hdf5::attribute::Hdf5Attribute],
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

fn read_voxels(
    file: &Hdf5File<std::fs::File>,
    data_address: u64,
    datatype: &consus_core::Datatype,
    total_elements: usize,
    total_bytes: usize,
    scaling: Option<&IntegerScaling>,
) -> Result<Vec<f32>> {
    let element_size = datatype
        .element_size()
        .context("Variable-length MINC2 image datatype is unsupported")?;
    let chunk_capacity = VOXEL_READ_BYTES / element_size * element_size;
    if chunk_capacity == 0 {
        bail!("MINC2 voxel element width {element_size} exceeds read scratch capacity");
    }
    let mut scratch = [0_u8; VOXEL_READ_BYTES];
    let mut output = Vec::new();
    let mut byte_offset = 0_usize;
    while byte_offset < total_bytes {
        let chunk_bytes = (total_bytes - byte_offset).min(chunk_capacity);
        let destination = &mut scratch[..chunk_bytes];
        let file_offset = u64::try_from(byte_offset).context("MINC2 read offset exceeds u64")?;
        file.read_contiguous_dataset_bytes(data_address, file_offset, destination)
            .map_err(|error| {
                anyhow::anyhow!(
                    "Failed to read MINC2 voxel data at byte offset {byte_offset}: {error}"
                )
            })?;
        let start_index = output.len();
        decode_raw_bytes_into(destination, datatype, start_index, &mut output, scaling)
            .with_context(|| format!("Decode MINC2 voxel chunk at element {start_index}"))?;
        byte_offset = byte_offset
            .checked_add(chunk_bytes)
            .context("MINC2 read offset overflows usize")?;
    }
    if output.len() != total_elements {
        bail!(
            "MINC2 voxel payload decoded {} elements, expected {total_elements}",
            output.len()
        );
    }
    Ok(output)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Read a MINC2 (.mnc / .mnc2) file into a 3-D `Image`.
///
/// # Arguments
///
/// - `path`: filesystem path to the MINC2 HDF5 file.
/// - `backend`: Coeus compute backend used for tensor allocation.
///
/// # Errors
///
/// Returns `Err` when:
/// - The file cannot be opened or is not valid HDF5.
/// - The required MINC2 HDF5 structure is missing or malformed.
/// - The image dataset uses chunked storage (not yet supported).
/// - Integer scaling metadata or a stored sample violates the MINC2 contract.
/// - A data type conversion fails.
pub fn read_minc<B, P>(path: P, backend: &B) -> Result<ritk_image::Image<f32, B, 3>>
where
    B: coeus_core::ComputeBackend,
    P: AsRef<Path>,
{
    let DecodedMinc {
        data,
        dims,
        origin,
        spacing,
        direction,
    } = decode_minc(path)?;
    ritk_image::Image::from_flat_on(data, dims, origin, spacing, direction, backend)
}

/// Backend-agnostic decoded MINC2 volume: voxels plus derived physical metadata.
/// Shared by format validation and native image construction.
struct DecodedMinc {
    data: Vec<f32>,
    dims: [usize; 3],
    origin: ritk_spatial::Point<3>,
    spacing: ritk_spatial::Spacing<3>,
    direction: ritk_spatial::Direction<3>,
}

fn decode_minc<P: AsRef<Path>>(path: P) -> Result<DecodedMinc> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open MINC2 file {:?}", path))?;
    let hdf5 = Hdf5File::open(file)
        .map_err(|e| anyhow::anyhow!("HDF5 open failed for {:?}: {}", path, e))?;

    let dimensions = read_dimension_metadata(&hdf5)
        .with_context(|| format!("Failed to read dimension metadata from {:?}", path))?;

    let image_addr = hdf5
        .open_path(IMAGE_PATH)
        .map_err(|e| anyhow::anyhow!("Cannot locate {}: {}", IMAGE_PATH, e))?;
    let dataset = hdf5
        .dataset_at(image_addr)
        .map_err(|e| anyhow::anyhow!("Cannot read image dataset metadata: {}", e))?;

    let image_attrs = hdf5
        .attributes_at(image_addr)
        .map_err(|e| anyhow::anyhow!("Cannot read image attributes: {}", e))?;
    let dimorder = read_dimorder(&image_attrs)?;

    if dataset.layout != StorageLayout::Contiguous {
        bail!(
            "MINC2 image dataset uses {:?} storage; only Contiguous is currently supported",
            dataset.layout
        );
    }

    let ordered_dims = order_dimensions_by_dimorder(&dimensions, &dimorder)?;
    let (origin, spacing, direction) = build_spatial_metadata(&ordered_dims);

    let shape_arr: [usize; 3] = [
        ordered_dims[0].length,
        ordered_dims[1].length,
        ordered_dims[2].length,
    ];

    let dataset_shape = dataset.shape.current_dims().to_vec();
    if dataset_shape.as_slice() != shape_arr {
        bail!(
            "Shape mismatch: dimorder dimensions give {shape_arr:?}, dataset has {dataset_shape:?}"
        );
    }
    let total_elements = checked_product(&dataset_shape, "MINC2 dataset")?;
    let expected_elements = checked_product(&shape_arr, "MINC2 dimension metadata")?;
    if expected_elements != total_elements {
        bail!(
            "Shape mismatch: dimorder dimensions give {} elements, dataset has {}",
            expected_elements,
            total_elements
        );
    }

    let element_size = dataset
        .datatype
        .element_size()
        .context("Variable-length MINC2 image datatype is unsupported")?;
    let total_bytes = total_elements
        .checked_mul(element_size)
        .context("MINC2 voxel data size overflows usize")?;
    let data_address = dataset
        .data_address
        .context("MINC2 image dataset has no contiguous data address")?;

    let default_valid_range = default_integer_valid_range(&dataset.datatype)?;
    let scaling = match default_valid_range {
        None => None,
        Some(default_valid_range) => {
            let valid_range = read_valid_range(&image_attrs, default_valid_range)?;
            let (image_minima, image_maxima) = read_image_ranges(&hdf5, shape_arr[0])?;
            let slice_length = shape_arr[1]
                .checked_mul(shape_arr[2])
                .context("MINC2 slice element count overflows usize")?;
            Some(IntegerScaling::new(
                valid_range,
                default_valid_range,
                image_minima,
                image_maxima,
                slice_length,
                total_elements,
            )?)
        }
    };
    let f32_data = read_voxels(
        &hdf5,
        data_address,
        &dataset.datatype,
        total_elements,
        total_bytes,
        scaling.as_ref(),
    )?;

    Ok(DecodedMinc {
        data: f32_data,
        dims: shape_arr,
        origin,
        spacing,
        direction,
    })
}

/// Backend-bound MINC2 reader.
pub struct MincReader<B: coeus_core::ComputeBackend> {
    backend: B,
}

impl<B: coeus_core::ComputeBackend> MincReader<B> {
    /// Construct a reader that creates images on `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Read a MINC2 file into a 3-D image using the stored backend.
    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<ritk_image::Image<f32, B, 3>> {
        read_minc(path, &self.backend)
    }
}
