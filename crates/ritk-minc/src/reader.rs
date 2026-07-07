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
//! 5. Read raw voxel bytes from contiguous storage and convert to `f32`.
//! 6. Construct `Image<B, 3>` with spatial metadata derived from
//!    dimension attributes and the dimorder axis mapping.
//!
//! # Contiguous Storage Requirement
//!
//! The current implementation reads contiguously-stored datasets only.
//! Chunked datasets require B-tree traversal and per-chunk decompression
//! which will be added in a follow-up sprint.

use crate::{
    convert::decode_raw_bytes,
    spatial::{
        build_spatial_metadata, order_dimensions_by_dimorder, read_dimension_metadata,
        read_dimorder,
    },
    IMAGE_PATH,
};
use anyhow::{bail, Context, Result};
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use ritk_core::image::Image;
use ritk_image::tensor::backend::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use std::path::Path;

// ── Public API ────────────────────────────────────────────────────────────────

/// Read a MINC2 (.mnc / .mnc2) file into a 3-D `Image`.
///
/// # Arguments
///
/// - `path`: filesystem path to the MINC2 HDF5 file.
/// - `device`: Burn backend device for tensor allocation.
///
/// # Errors
///
/// Returns `Err` when:
/// - The file cannot be opened or is not valid HDF5.
/// - The required MINC2 HDF5 structure is missing or malformed.
/// - The image dataset uses chunked storage (not yet supported).
/// - A data type conversion fails.
pub fn read_minc<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
    let DecodedMinc {
        data,
        dims,
        origin,
        spacing,
        direction,
    } = decode_minc(path)?;
    let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), device);
    Ok(Image::new(tensor, origin, spacing, direction))
}

/// Backend-agnostic decoded MINC2 volume: voxels plus derived physical metadata.
/// Shared by the Burn and Coeus reader paths.
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

    let data_address = dataset
        .data_address
        .ok_or_else(|| anyhow::anyhow!("MINC2 image dataset has no contiguous data address"))?;

    let elem_size = dataset
        .datatype
        .element_size()
        .ok_or_else(|| anyhow::anyhow!("Variable-length image datatype not supported"))?;
    let total_elements = dataset.shape.num_elements();
    let total_bytes = total_elements
        .checked_mul(elem_size)
        .ok_or_else(|| anyhow::anyhow!("Voxel data size overflow"))?;

    // Bound the speculative allocation: `total_bytes` derives from the dataset
    // shape and may exceed the bytes actually backed on disk. `read_bounded_with`
    // grows the buffer per confirmed chunk, so a hostile shape surfaces the
    // HDF5 read error rather than reserving the full claimed size up front.
    let raw = ritk_core::io_bounds::read_bounded_with(total_bytes, |offset, sub| {
        hdf5.read_contiguous_dataset_bytes(data_address, offset, sub)
    })
    .map_err(|e| anyhow::anyhow!("Failed to read voxel data: {}", e))?;

    let f32_data = decode_raw_bytes(&raw, &dataset.datatype)?;

    let ordered_dims = order_dimensions_by_dimorder(&dimensions, &dimorder)?;
    let (origin, spacing, direction) = build_spatial_metadata(&ordered_dims);

    let shape_arr: [usize; 3] = [
        ordered_dims[0].length,
        ordered_dims[1].length,
        ordered_dims[2].length,
    ];

    let expected_elements: usize = shape_arr.iter().product();
    if expected_elements != total_elements {
        bail!(
            "Shape mismatch: dimorder dimensions give {} elements, dataset has {}",
            expected_elements,
            total_elements
        );
    }

    Ok(DecodedMinc {
        data: f32_data,
        dims: shape_arr,
        origin,
        spacing,
        direction,
    })
}

/// Typed reader wrapping `read_minc` for API consistency.
///
/// Carries a backend device so it can implement `ImageReader<B, 3>`
/// without requiring callers to pass a device at read time.
pub struct MincReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MincReader<B> {
    /// Construct a `MincReader` bound to the given backend device.
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    /// Read a MINC2 file into a 3-D image using the stored device.
    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Image<B, 3>> {
        read_minc(path, &self.device)
    }
}

/// Atlas-native-substrate entry points (transitional module: plain
/// end-state names, disambiguated from the Burn functions by module
/// path only; folds away when the Burn path is deleted — ADR 0002 A1).
pub mod native {
    #[allow(unused_imports)]
    use super::*;

    /// Read a MINC2 file into a Coeus-backed 3-D image on `backend`.
    ///
    /// The Atlas-tensor counterpart to [`read_minc`]: shares the HDF5 navigation,
    /// bounded contiguous voxel read, decode, and geometry derivation with the Burn
    /// path via `decode_minc`, differing only in the final image construction.
    pub fn read_minc<B, P>(path: P, backend: &B) -> Result<ritk_image::native::Image<f32, B, 3>>
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
        ritk_image::native::Image::from_flat_on(data, dims, origin, spacing, direction, backend)
    }
}
