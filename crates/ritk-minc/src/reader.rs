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
//! 5. For an integer image, validate `valid_range` and the scalar or per-slice
//!    `image-min` / `image-max` real ranges into one real-value map per slice
//!    (expanded to one per slice only after the voxel payload is read).
//! 6. Read the voxels in the stored sample type in bounded steps and reject any
//!    integer sample outside `valid_range`.
//! 7. Convert the stored samples to the requested type under the caller's
//!    [`Conversion`], apply each slice's map in that type, and construct
//!    `Image<T, B, 3>` with spatial metadata derived from the dimension
//!    attributes and the dimorder axis mapping.
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
    datatype::stored_type,
    image_ranges::{read_image_ranges, read_valid_range},
    payload::read_payload,
    real_map::RealValueMap,
    scaling::{integer_storage_range, IntegerScaling},
    spatial::{
        build_spatial_metadata, order_dimensions_by_dimorder, read_dimension_metadata,
        read_dimorder,
    },
    IMAGE_PATH,
};
use anyhow::{bail, Context, Result};
use coeus_core::ComputeBackend;
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use eunomia::NumericElement;
use ritk_codecs::sample::{Conversion, Sample};
use std::path::Path;

fn checked_product(values: &[usize], label: &str) -> Result<usize> {
    values.iter().copied().try_fold(1_usize, |product, value| {
        product
            .checked_mul(value)
            .with_context(|| format!("{label} element count overflows usize"))
    })
}

/// Read a MINC2 (.mnc / .mnc2) file into a 3-D `Image` of `T` holding real
/// intensities.
///
/// The stored samples (`u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`,
/// `f32`, or `f64`) convert to `T` under `conversion`
/// ([`Exact`](ritk_codecs::sample::Exact) refuses any conversion that could
/// change a value). An integer image then maps each slice from its
/// `valid_range` to its `image-min` / `image-max` real range, in `T`'s
/// arithmetic: `real = (stored - valid_min) * slope + image_min` with
/// `slope = (image_max - image_min) / (valid_max - valid_min)`
/// ([`RealValueMap`]). A floating-point image bypasses that map. A file that stores no `image-min` / `image-max` maps to the MINC
/// default real range `[0, 1]`.
///
/// # Arguments
///
/// - `path`: filesystem path to the MINC2 HDF5 file.
/// - `backend`: Coeus compute backend used for tensor allocation.
/// - `conversion`: how the stored samples become `T`.
///
/// # Errors
///
/// Returns `Err` when:
/// - The file cannot be opened or is not valid HDF5.
/// - The required MINC2 HDF5 structure is missing or malformed.
/// - The image dataset uses chunked storage (not yet supported).
/// - Integer scaling metadata or a stored sample violates the MINC2 contract.
/// - `conversion` refuses the stored type.
/// - A slice's map is not the identity and `T` is an integer type, or the map
///   leaves the range of `T` — use [`read_minc_stored`] for the stored samples
///   and the maps.
pub fn read_minc<T, C, B, P>(
    path: P,
    backend: &B,
    conversion: C,
) -> Result<ritk_image::Image<T, B, 3>>
where
    T: Sample,
    C: Conversion,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let mut decoded = decode_minc::<T, C>(path.as_ref(), conversion)?;
    apply_slice_maps(&mut decoded.data, &decoded.maps, decoded.slice_length)?;
    decoded.into_image(backend)
}

/// Read a MINC2 file as its stored samples in `T`, with each slice's real-value
/// map left unapplied.
///
/// The returned [`RealValueMap`]s hold one map per slice along the first
/// spatial axis, mapping the stored samples to real intensities (the identity
/// for a floating-point image). Reading in the stored type keeps every sample
/// exact.
///
/// # Errors
///
/// Returns the errors of [`read_minc`] except the map refusals.
pub fn read_minc_stored<T, C, B, P>(
    path: P,
    backend: &B,
    conversion: C,
) -> Result<(ritk_image::Image<T, B, 3>, Vec<RealValueMap>)>
where
    T: Sample,
    C: Conversion,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let mut decoded = decode_minc::<T, C>(path.as_ref(), conversion)?;
    let maps = std::mem::take(&mut decoded.maps);
    let image = decoded.into_image(backend)?;
    Ok((image, maps))
}

/// Backend-agnostic decoded MINC2 volume: stored samples converted to `T`,
/// the unapplied per-slice maps, and the derived physical metadata.
struct DecodedMinc<T> {
    data: Vec<T>,
    maps: Vec<RealValueMap>,
    slice_length: usize,
    dims: [usize; 3],
    origin: ritk_spatial::Point<3>,
    spacing: ritk_spatial::Spacing<3>,
    direction: ritk_spatial::Direction<3>,
}

impl<T: Sample> DecodedMinc<T> {
    fn into_image<B: ComputeBackend>(self, backend: &B) -> Result<ritk_image::Image<T, B, 3>> {
        ritk_image::Image::from_flat_on(
            self.data,
            self.dims,
            self.origin,
            self.spacing,
            self.direction,
            backend,
        )
    }
}

/// Map each slice of `data` by its [`RealValueMap`] in `T`.
fn apply_slice_maps<T: Sample>(
    data: &mut [T],
    maps: &[RealValueMap],
    slice_length: usize,
) -> Result<()> {
    if slice_length == 0 {
        return Ok(());
    }
    for (index, (slice, map)) in data.chunks_exact_mut(slice_length).zip(maps).enumerate() {
        if map.is_identity() {
            continue;
        }
        map.apply(slice).with_context(|| {
            format!(
                "MINC2 slice {index} maps stored values by (x - {}) * {} + {}; \
                 read_minc_stored returns the stored samples and the maps",
                map.valid_minimum(),
                map.slope(),
                map.intercept()
            )
        })?;
        if let Some(offset) = slice
            .iter()
            .position(|&value| !NumericElement::to_f64(value).is_finite())
        {
            bail!(
                "MINC2 scaled voxel {} leaves the finite range of {}: the mapped value or its \
                 intermediate (x - valid_min) * slope is not representable in {}; \
                 read into a wider floating-point type",
                index * slice_length + offset,
                T::TYPE,
                T::TYPE
            );
        }
    }
    Ok(())
}

fn decode_minc<T: Sample, C: Conversion>(path: &Path, conversion: C) -> Result<DecodedMinc<T>> {
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

    let stored = stored_type(&dataset.datatype)?;
    let data_address = dataset
        .data_address
        .context("MINC2 image dataset has no contiguous data address")?;

    let slice_length = shape_arr[1]
        .checked_mul(shape_arr[2])
        .context("MINC2 slice element count overflows usize")?;
    let scaling = match integer_storage_range(stored.sample_type) {
        None => None,
        Some(storage_range) => {
            let valid_range = read_valid_range(&image_attrs, storage_range)?;
            let (image_minima, image_maxima) = read_image_ranges(&hdf5, shape_arr[0])?;
            Some(IntegerScaling::new(
                valid_range,
                storage_range,
                &image_minima,
                &image_maxima,
                slice_length,
                total_elements,
            )?)
        }
    };
    let samples = read_payload(&hdf5, data_address, stored, total_elements)?;
    let maps = match &scaling {
        Some(scaling) => {
            scaling.check_stored(&samples)?;
            scaling.slice_maps()
        }
        None => vec![RealValueMap::IDENTITY; shape_arr[0]],
    };
    let data = conversion.convert::<T>(samples)?;

    Ok(DecodedMinc {
        data,
        maps,
        slice_length,
        dims: shape_arr,
        origin,
        spacing,
        direction,
    })
}

/// Backend-bound MINC2 reader.
pub struct MincReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> MincReader<B> {
    /// Construct a reader that creates images on `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Read a MINC2 file into a 3-D image of `T` using the stored backend.
    ///
    /// # Errors
    ///
    /// Returns the error of [`read_minc`].
    pub fn read<T: Sample, C: Conversion, P: AsRef<Path>>(
        &self,
        path: P,
        conversion: C,
    ) -> Result<ritk_image::Image<T, B, 3>> {
        read_minc(path, &self.backend, conversion)
    }
}
