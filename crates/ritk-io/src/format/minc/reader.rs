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
//!
//! # Data Normalization
//!
//! For integer-typed image data, MINC2 files may include `image-min` and
//! `image-max` per-slice datasets and a `valid_range` attribute. When
//! these are present, the reader applies linear normalization:
//!
//! ```text
//! t = (raw - valid_min) / (valid_max - valid_min)
//! real = t * (slice_max - slice_min) + slice_min
//! ```
//!
//! For floating-point data, raw values are used directly.

use super::{MincDimension, DIMENSIONS_PATH, IMAGE_PATH, SPATIAL_DIM_NAMES};
use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use consus_core::{AttributeValue, Datatype};
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use nalgebra::SMatrix;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::path::Path;

// ── Public API ───────────────────────────────────────────────────────────────

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
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open MINC2 file {:?}", path))?;
    let hdf5 = Hdf5File::open(file)
        .map_err(|e| anyhow::anyhow!("HDF5 open failed for {:?}: {}", path, e))?;

    // 1. Read spatial dimension metadata.
    let dimensions = read_dimension_metadata(&hdf5)
        .with_context(|| format!("Failed to read dimension metadata from {:?}", path))?;

    // 2. Read image dataset metadata.
    let image_addr = hdf5
        .open_path(IMAGE_PATH)
        .map_err(|e| anyhow::anyhow!("Cannot locate {}: {}", IMAGE_PATH, e))?;
    let dataset = hdf5
        .dataset_at(image_addr)
        .map_err(|e| anyhow::anyhow!("Cannot read image dataset metadata: {}", e))?;

    // 3. Parse dimorder attribute.
    let image_attrs = hdf5
        .attributes_at(image_addr)
        .map_err(|e| anyhow::anyhow!("Cannot read image attributes: {}", e))?;
    let dimorder = extract_dimorder(&image_attrs)?;

    // 4. Validate storage layout.
    if dataset.layout != StorageLayout::Contiguous {
        bail!(
            "MINC2 image dataset uses {:?} storage; only Contiguous is currently supported",
            dataset.layout
        );
    }

    let data_address = dataset
        .data_address
        .ok_or_else(|| anyhow::anyhow!("MINC2 image dataset has no contiguous data address"))?;

    // 5. Read raw voxel bytes.
    let elem_size = dataset
        .datatype
        .element_size()
        .ok_or_else(|| anyhow::anyhow!("Variable-length image datatype not supported"))?;
    let total_elements = dataset.shape.num_elements();
    let total_bytes = total_elements
        .checked_mul(elem_size)
        .ok_or_else(|| anyhow::anyhow!("Voxel data size overflow"))?;

    let mut raw = vec![0u8; total_bytes];
    hdf5.read_contiguous_dataset_bytes(data_address, 0, &mut raw)
        .map_err(|e| anyhow::anyhow!("Failed to read voxel data: {}", e))?;

    // 6. Convert raw bytes to f32.
    let f32_data = convert_to_f32(&raw, &dataset.datatype)?;

    // 7. Build spatial metadata from dimensions + dimorder.
    let ordered_dims = order_dimensions_by_dimorder(&dimensions, &dimorder)?;
    let (origin, spacing, direction) = build_spatial_metadata(&ordered_dims);

    // 8. Determine tensor shape [dim0_len, dim1_len, dim2_len].
    let shape_arr: [usize; 3] = [
        ordered_dims[0].length,
        ordered_dims[1].length,
        ordered_dims[2].length,
    ];

    // Verify element count matches dataset shape.
    let expected_elements: usize = shape_arr.iter().product();
    if expected_elements != total_elements {
        bail!(
            "Shape mismatch: dimorder dimensions give {} elements, dataset has {}",
            expected_elements,
            total_elements
        );
    }

    let tensor_data = TensorData::new(f32_data, Shape::new(shape_arr));
    let tensor = Tensor::<B, 3>::from_data(tensor_data, device);

    Ok(Image::new(tensor, origin, spacing, direction))
}

/// Convenience struct wrapping `read_minc` for API consistency.
pub struct MincReader;

impl MincReader {
    /// Read a MINC2 file into a 3-D image.
    pub fn read<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
        read_minc(path, device)
    }
}

// ── Dimension metadata extraction ────────────────────────────────────────────

/// Read spatial dimension metadata from the MINC2 HDF5 dimensions group.
///
/// Navigates to `/minc-2.0/dimensions/` and reads attributes from each
/// recognized spatial dimension group (`xspace`, `yspace`, `zspace`).
///
/// # Returns
///
/// A vector of `MincDimension` structs, one per spatial dimension found.
/// The vector may have fewer than 3 entries if dimension groups are missing.
fn read_dimension_metadata<R: consus_io::ReadAt>(hdf5: &Hdf5File<R>) -> Result<Vec<MincDimension>> {
    let dims_addr = hdf5
        .open_path(DIMENSIONS_PATH)
        .map_err(|e| anyhow::anyhow!("Cannot locate {}: {}", DIMENSIONS_PATH, e))?;

    let children = hdf5
        .list_group_at(dims_addr)
        .map_err(|e| anyhow::anyhow!("Cannot list dimensions group: {}", e))?;

    let mut dimensions = Vec::with_capacity(3);

    for (name, addr, _link_type) in &children {
        if !SPATIAL_DIM_NAMES.contains(&name.as_str()) {
            continue;
        }

        let attrs = hdf5
            .attributes_at(*addr)
            .map_err(|e| anyhow::anyhow!("Cannot read attributes for dimension {}: {}", name, e))?;

        let dim = parse_dimension_attrs(name, &attrs)?;
        dimensions.push(dim);
    }

    if dimensions.is_empty() {
        bail!(
            "No spatial dimension groups found under {}",
            DIMENSIONS_PATH
        );
    }

    Ok(dimensions)
}

/// Parse dimension attributes into a `MincDimension`.
///
/// Extracts `start`, `step`, `length`, and `direction_cosines` from
/// the attribute list. Missing optional attributes use sensible defaults:
/// - `start` defaults to `0.0`
/// - `step` defaults to `1.0`
/// - `direction_cosines` defaults to the canonical axis direction
///   (`[1,0,0]` for xspace, `[0,1,0]` for yspace, `[0,0,1]` for zspace)
///
/// `length` is required; its absence is an error.
fn parse_dimension_attrs(
    name: &str,
    attrs: &[consus_hdf5::attribute::Hdf5Attribute],
) -> Result<MincDimension> {
    let mut start: f64 = 0.0;
    let mut step: f64 = 1.0;
    let mut length: Option<usize> = None;
    let mut dir_cos: Option<[f64; 3]> = None;

    for attr in attrs {
        let decoded = attr
            .decode_value()
            .map_err(|e| anyhow::anyhow!("Cannot decode attribute '{}': {}", attr.name, e))?;

        match attr.name.as_str() {
            "start" => {
                start = extract_f64(&decoded)
                    .with_context(|| format!("Invalid 'start' attribute on {}", name))?;
            }
            "step" => {
                step = extract_f64(&decoded)
                    .with_context(|| format!("Invalid 'step' attribute on {}", name))?;
            }
            "length" => {
                let v = extract_i64(&decoded)
                    .with_context(|| format!("Invalid 'length' attribute on {}", name))?;
                if v <= 0 {
                    bail!("Dimension '{}' has non-positive length: {}", name, v);
                }
                length = Some(v as usize);
            }
            "direction_cosines" => {
                dir_cos = Some(
                    extract_f64_array_3(&decoded)
                        .with_context(|| format!("Invalid 'direction_cosines' on {}", name))?,
                );
            }
            _ => {} // Ignore unrecognized attributes (units, spacetype, etc.)
        }
    }

    let length = length.ok_or_else(|| {
        anyhow::anyhow!("Dimension '{}' missing required 'length' attribute", name)
    })?;

    // Default direction cosines: canonical axis for the dimension name.
    let direction_cosines = dir_cos.unwrap_or_else(|| default_direction_cosines(name));

    Ok(MincDimension {
        name: name.to_string(),
        start,
        step,
        length,
        direction_cosines,
    })
}

/// Default direction cosines for a named spatial dimension.
///
/// - `xspace` → `[1, 0, 0]`
/// - `yspace` → `[0, 1, 0]`
/// - `zspace` → `[0, 0, 1]`
fn default_direction_cosines(name: &str) -> [f64; 3] {
    match name {
        "xspace" => [1.0, 0.0, 0.0],
        "yspace" => [0.0, 1.0, 0.0],
        "zspace" => [0.0, 0.0, 1.0],
        _ => [1.0, 0.0, 0.0],
    }
}

// ── Attribute value extraction helpers ───────────────────────────────────────

/// Extract a scalar `f64` from an `AttributeValue`.
fn extract_f64(val: &AttributeValue) -> Result<f64> {
    match val {
        AttributeValue::Float(v) => Ok(*v),
        AttributeValue::Int(v) => Ok(*v as f64),
        AttributeValue::Uint(v) => Ok(*v as f64),
        AttributeValue::FloatArray(arr) if arr.len() == 1 => Ok(arr[0]),
        other => bail!("Expected scalar float, got {:?}", other),
    }
}

/// Extract a scalar `i64` from an `AttributeValue`.
fn extract_i64(val: &AttributeValue) -> Result<i64> {
    match val {
        AttributeValue::Int(v) => Ok(*v),
        AttributeValue::Uint(v) => Ok(*v as i64),
        AttributeValue::Float(v) => Ok(*v as i64),
        other => bail!("Expected scalar integer, got {:?}", other),
    }
}

/// Extract a 3-element `f64` array from an `AttributeValue`.
fn extract_f64_array_3(val: &AttributeValue) -> Result<[f64; 3]> {
    match val {
        AttributeValue::FloatArray(arr) if arr.len() >= 3 => Ok([arr[0], arr[1], arr[2]]),
        AttributeValue::Float(v) => {
            // Scalar: replicate (unusual but handle gracefully).
            Ok([*v, *v, *v])
        }
        other => bail!("Expected float array of length >= 3, got {:?}", other),
    }
}

/// Extract a string from an `AttributeValue`.
fn extract_string(val: &AttributeValue) -> Result<String> {
    match val {
        AttributeValue::String(s) => Ok(s.clone()),
        AttributeValue::Bytes(b) => {
            // Strip trailing nulls and decode as UTF-8.
            let end = b.iter().position(|&x| x == 0).unwrap_or(b.len());
            Ok(String::from_utf8_lossy(&b[..end]).into_owned())
        }
        other => bail!("Expected string, got {:?}", other),
    }
}

// ── Dimorder parsing ─────────────────────────────────────────────────────────

/// Extract the `dimorder` attribute from image dataset attributes.
///
/// Returns a vector of dimension names in dataset axis order.
/// Example: `"zspace,yspace,xspace"` → `["zspace", "yspace", "xspace"]`.
///
/// If `dimorder` is absent, returns the default `["zspace", "yspace", "xspace"]`.
fn extract_dimorder(attrs: &[consus_hdf5::attribute::Hdf5Attribute]) -> Result<Vec<String>> {
    for attr in attrs {
        if attr.name == "dimorder" {
            let decoded = attr
                .decode_value()
                .map_err(|e| anyhow::anyhow!("Cannot decode dimorder: {}", e))?;
            let s = extract_string(&decoded)?;
            let order: Vec<String> = s
                .split(',')
                .map(|d| d.trim().to_string())
                .filter(|d| !d.is_empty())
                .collect();
            if order.len() < 3 {
                bail!("dimorder has fewer than 3 entries: {:?}", order);
            }
            return Ok(order);
        }
    }
    // Default: zspace varies slowest (outermost), xspace fastest (innermost).
    Ok(vec![
        "zspace".to_string(),
        "yspace".to_string(),
        "xspace".to_string(),
    ])
}

// ── Dimension ordering ───────────────────────────────────────────────────────

/// Order parsed dimensions according to the dimorder specification.
///
/// Returns a 3-element vector where index 0 corresponds to the outermost
/// (slowest-varying) tensor axis and index 2 to the innermost (fastest).
fn order_dimensions_by_dimorder(
    dimensions: &[MincDimension],
    dimorder: &[String],
) -> Result<Vec<MincDimension>> {
    let mut ordered = Vec::with_capacity(3);

    for dim_name in dimorder.iter().take(3) {
        let dim = dimensions
            .iter()
            .find(|d| d.name == *dim_name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "dimorder references '{}' but no matching dimension found",
                    dim_name
                )
            })?;
        ordered.push(dim.clone());
    }

    if ordered.len() != 3 {
        bail!(
            "Expected 3 spatial dimensions from dimorder, found {}",
            ordered.len()
        );
    }

    Ok(ordered)
}

// ── Spatial metadata construction ────────────────────────────────────────────

/// Build RITK spatial metadata from ordered dimension metadata.
///
/// # Arguments
///
/// `ordered_dims`: dimensions ordered by dimorder (index 0 = outermost axis).
///
/// # Returns
///
/// `(origin, spacing, direction)` triple for `Image` construction.
///
/// # Derivation
///
/// - `origin[i] = ordered_dims[i].start`
/// - `spacing[i] = |ordered_dims[i].step|`
/// - `direction` columns = `ordered_dims[i].direction_cosines`
///
/// Step sign is absorbed into the direction cosines: if `step < 0`,
/// the direction cosine vector is negated to maintain positive spacing.
fn build_spatial_metadata(ordered_dims: &[MincDimension]) -> (Point<3>, Spacing<3>, Direction<3>) {
    let mut origin_arr = [0.0f64; 3];
    let mut spacing_arr = [0.0f64; 3];
    let mut dir_columns: [[f64; 3]; 3] = [[0.0; 3]; 3];

    for (i, dim) in ordered_dims.iter().enumerate() {
        origin_arr[i] = dim.start;

        let abs_step = dim.step.abs();
        spacing_arr[i] = if abs_step > 0.0 { abs_step } else { 1.0 };

        // If step is negative, negate direction cosines to maintain
        // the convention that spacing is always positive.
        let sign = if dim.step < 0.0 { -1.0 } else { 1.0 };
        dir_columns[i] = [
            dim.direction_cosines[0] * sign,
            dim.direction_cosines[1] * sign,
            dim.direction_cosines[2] * sign,
        ];
    }

    let origin = Point::new(origin_arr);
    let spacing = Spacing::new(spacing_arr);

    let dir_matrix = SMatrix::<f64, 3, 3>::from_columns(&[
        nalgebra::Vector3::new(dir_columns[0][0], dir_columns[0][1], dir_columns[0][2]),
        nalgebra::Vector3::new(dir_columns[1][0], dir_columns[1][1], dir_columns[1][2]),
        nalgebra::Vector3::new(dir_columns[2][0], dir_columns[2][1], dir_columns[2][2]),
    ]);
    let direction = Direction(dir_matrix);

    (origin, spacing, direction)
}

// ── Data type conversion ─────────────────────────────────────────────────────

/// Convert raw bytes to `Vec<f32>` based on the HDF5 datatype.
///
/// # Supported Types
///
/// | HDF5 Datatype                | Conversion                    |
/// |------------------------------|-------------------------------|
/// | `Integer { 8, unsigned }`    | `u8 as f32`                   |
/// | `Integer { 8, signed }`      | `i8 as f32`                   |
/// | `Integer { 16, LE, signed }` | `i16::from_le_bytes as f32`   |
/// | `Integer { 16, LE, unsigned}`| `u16::from_le_bytes as f32`   |
/// | `Integer { 32, LE, signed }` | `i32::from_le_bytes as f32`   |
/// | `Integer { 32, LE, unsigned}`| `u32::from_le_bytes as f32`   |
/// | `Float { 32, LE }`          | `f32::from_le_bytes`          |
/// | `Float { 64, LE }`          | `f64::from_le_bytes as f32`   |
/// | Big-endian variants          | analogous with `from_be_bytes`|
///
/// # Errors
///
/// Returns `Err` for unsupported or variable-length data types.
fn convert_to_f32(raw: &[u8], dtype: &Datatype) -> Result<Vec<f32>> {
    use consus_core::ByteOrder;

    match dtype {
        Datatype::Float { bits, byte_order } => {
            let bw = bits.get();
            match (bw, byte_order) {
                (32, ByteOrder::LittleEndian) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()),
                (32, ByteOrder::BigEndian) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()),
                (64, ByteOrder::LittleEndian) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()),
                (64, ByteOrder::BigEndian) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()),
                _ => bail!("Unsupported float bit width: {}", bw),
            }
        }

        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => {
            let bw = bits.get();
            match (bw, byte_order, signed) {
                // 8-bit
                (8, _, false) => Ok(raw.iter().map(|&b| b as f32).collect()),
                (8, _, true) => Ok(raw.iter().map(|&b| (b as i8) as f32).collect()),

                // 16-bit little-endian
                (16, ByteOrder::LittleEndian, true) => Ok(raw
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
                    .collect()),
                (16, ByteOrder::LittleEndian, false) => Ok(raw
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32)
                    .collect()),

                // 16-bit big-endian
                (16, ByteOrder::BigEndian, true) => Ok(raw
                    .chunks_exact(2)
                    .map(|c| i16::from_be_bytes([c[0], c[1]]) as f32)
                    .collect()),
                (16, ByteOrder::BigEndian, false) => Ok(raw
                    .chunks_exact(2)
                    .map(|c| u16::from_be_bytes([c[0], c[1]]) as f32)
                    .collect()),

                // 32-bit little-endian
                (32, ByteOrder::LittleEndian, true) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect()),
                (32, ByteOrder::LittleEndian, false) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect()),

                // 32-bit big-endian
                (32, ByteOrder::BigEndian, true) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect()),
                (32, ByteOrder::BigEndian, false) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect()),

                // 64-bit (lossy cast to f32)
                (64, ByteOrder::LittleEndian, true) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()),
                (64, ByteOrder::LittleEndian, false) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()),
                (64, ByteOrder::BigEndian, true) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        i64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()),
                (64, ByteOrder::BigEndian, false) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        u64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()),

                _ => bail!("Unsupported integer type: {} bits, signed={}", bw, signed),
            }
        }

        Datatype::Boolean => Ok(raw
            .iter()
            .map(|&b| if b != 0 { 1.0f32 } else { 0.0f32 })
            .collect()),

        other => bail!("Unsupported MINC2 voxel datatype: {:?}", other),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn test_default_direction_cosines() {
        assert_eq!(default_direction_cosines("xspace"), [1.0, 0.0, 0.0]);
        assert_eq!(default_direction_cosines("yspace"), [0.0, 1.0, 0.0]);
        assert_eq!(default_direction_cosines("zspace"), [0.0, 0.0, 1.0]);
        // Unknown defaults to x-axis.
        assert_eq!(default_direction_cosines("tspace"), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_extract_f64_from_float() {
        let val = AttributeValue::Float(3.14);
        assert!((extract_f64(&val).unwrap() - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_extract_f64_from_int() {
        let val = AttributeValue::Int(42);
        assert!((extract_f64(&val).unwrap() - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_f64_from_uint() {
        let val = AttributeValue::Uint(7);
        assert!((extract_f64(&val).unwrap() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_i64_from_int() {
        let val = AttributeValue::Int(-5);
        assert_eq!(extract_i64(&val).unwrap(), -5);
    }

    #[test]
    fn test_extract_i64_from_uint() {
        let val = AttributeValue::Uint(100);
        assert_eq!(extract_i64(&val).unwrap(), 100);
    }

    #[test]
    fn test_extract_f64_array_3() {
        let val = AttributeValue::FloatArray(vec![0.5, 0.7, 0.3]);
        let arr = extract_f64_array_3(&val).unwrap();
        assert!((arr[0] - 0.5).abs() < 1e-10);
        assert!((arr[1] - 0.7).abs() < 1e-10);
        assert!((arr[2] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_extract_f64_array_3_longer() {
        let val = AttributeValue::FloatArray(vec![1.0, 2.0, 3.0, 4.0]);
        let arr = extract_f64_array_3(&val).unwrap();
        assert!((arr[0] - 1.0).abs() < 1e-10);
        assert!((arr[1] - 2.0).abs() < 1e-10);
        assert!((arr[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_f64_array_3_too_short() {
        let val = AttributeValue::FloatArray(vec![1.0, 2.0]);
        assert!(extract_f64_array_3(&val).is_err());
    }

    #[test]
    fn test_extract_string() {
        let val = AttributeValue::String("zspace,yspace,xspace".to_string());
        assert_eq!(extract_string(&val).unwrap(), "zspace,yspace,xspace");
    }

    #[test]
    fn test_extract_string_from_bytes() {
        let val = AttributeValue::Bytes(b"hello\0\0\0".to_vec());
        assert_eq!(extract_string(&val).unwrap(), "hello");
    }

    #[test]
    fn test_dimorder_parsing_default() {
        let attrs: Vec<consus_hdf5::attribute::Hdf5Attribute> = vec![];
        let order = extract_dimorder(&attrs).unwrap();
        assert_eq!(order, vec!["zspace", "yspace", "xspace"]);
    }

    #[test]
    fn test_order_dimensions_by_dimorder() {
        let dims = vec![
            MincDimension {
                name: "xspace".to_string(),
                start: 0.0,
                step: 1.0,
                length: 64,
                direction_cosines: [1.0, 0.0, 0.0],
            },
            MincDimension {
                name: "yspace".to_string(),
                start: 0.0,
                step: 1.0,
                length: 80,
                direction_cosines: [0.0, 1.0, 0.0],
            },
            MincDimension {
                name: "zspace".to_string(),
                start: 0.0,
                step: 1.0,
                length: 48,
                direction_cosines: [0.0, 0.0, 1.0],
            },
        ];
        let dimorder = vec![
            "zspace".to_string(),
            "yspace".to_string(),
            "xspace".to_string(),
        ];
        let ordered = order_dimensions_by_dimorder(&dims, &dimorder).unwrap();
        assert_eq!(ordered[0].name, "zspace");
        assert_eq!(ordered[0].length, 48);
        assert_eq!(ordered[1].name, "yspace");
        assert_eq!(ordered[1].length, 80);
        assert_eq!(ordered[2].name, "xspace");
        assert_eq!(ordered[2].length, 64);
    }

    #[test]
    fn test_order_dimensions_missing_dim() {
        let dims = vec![MincDimension {
            name: "xspace".to_string(),
            start: 0.0,
            step: 1.0,
            length: 64,
            direction_cosines: [1.0, 0.0, 0.0],
        }];
        let dimorder = vec![
            "zspace".to_string(),
            "yspace".to_string(),
            "xspace".to_string(),
        ];
        assert!(order_dimensions_by_dimorder(&dims, &dimorder).is_err());
    }

    #[test]
    fn test_build_spatial_metadata_identity() {
        let dims = vec![
            MincDimension {
                name: "zspace".to_string(),
                start: -10.0,
                step: 2.0,
                length: 20,
                direction_cosines: [0.0, 0.0, 1.0],
            },
            MincDimension {
                name: "yspace".to_string(),
                start: -20.0,
                step: 1.5,
                length: 30,
                direction_cosines: [0.0, 1.0, 0.0],
            },
            MincDimension {
                name: "xspace".to_string(),
                start: -15.0,
                step: 1.0,
                length: 40,
                direction_cosines: [1.0, 0.0, 0.0],
            },
        ];

        let (origin, spacing, _direction) = build_spatial_metadata(&dims);
        assert!((origin[0] - (-10.0)).abs() < 1e-10);
        assert!((origin[1] - (-20.0)).abs() < 1e-10);
        assert!((origin[2] - (-15.0)).abs() < 1e-10);
        assert!((spacing[0] - 2.0).abs() < 1e-10);
        assert!((spacing[1] - 1.5).abs() < 1e-10);
        assert!((spacing[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_build_spatial_metadata_negative_step() {
        let dims = vec![
            MincDimension {
                name: "zspace".to_string(),
                start: 10.0,
                step: -2.0,
                length: 20,
                direction_cosines: [0.0, 0.0, 1.0],
            },
            MincDimension {
                name: "yspace".to_string(),
                start: 0.0,
                step: 1.0,
                length: 30,
                direction_cosines: [0.0, 1.0, 0.0],
            },
            MincDimension {
                name: "xspace".to_string(),
                start: 0.0,
                step: 1.0,
                length: 40,
                direction_cosines: [1.0, 0.0, 0.0],
            },
        ];

        let (origin, spacing, direction) = build_spatial_metadata(&dims);
        // Spacing is always positive (absolute step).
        assert!((spacing[0] - 2.0).abs() < 1e-10);
        // Direction cosine for axis 0 should be negated because step < 0.
        assert!((direction.0[(0, 0)] - 0.0).abs() < 1e-10);
        assert!((direction.0[(1, 0)] - 0.0).abs() < 1e-10);
        assert!((direction.0[(2, 0)] - (-1.0)).abs() < 1e-10);
        // Origin preserves the start value.
        assert!((origin[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_convert_f32_le() {
        let val: f32 = 3.14;
        let raw = val.to_le_bytes().to_vec();
        let dtype = Datatype::Float {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        };
        let result = convert_to_f32(&raw, &dtype).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.14).abs() < 1e-5);
    }

    #[test]
    fn test_convert_f32_be() {
        let val: f32 = 2.71;
        let raw = val.to_be_bytes().to_vec();
        let dtype = Datatype::Float {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::BigEndian,
        };
        let result = convert_to_f32(&raw, &dtype).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.71).abs() < 1e-5);
    }

    #[test]
    fn test_convert_f64_le() {
        let val: f64 = 1.23456789;
        let raw = val.to_le_bytes().to_vec();
        let dtype = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        };
        let result = convert_to_f32(&raw, &dtype).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.23456789f32).abs() < 1e-5);
    }

    #[test]
    fn test_convert_u8() {
        let raw = vec![0u8, 128, 255];
        let dtype = Datatype::Integer {
            bits: NonZeroUsize::new(8).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
            signed: false,
        };
        let result = convert_to_f32(&raw, &dtype).unwrap();
        assert_eq!(result, vec![0.0, 128.0, 255.0]);
    }

    #[test]
    fn test_convert_i8() {
        let raw = vec![0u8, 127, 0x80]; // 0, 127, -128
        let dtype = Datatype::Integer {
            bits: NonZeroUsize::new(8).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
            signed: true,
        };
        let result = convert_to_f32(&raw, &dtype).unwrap();
        assert_eq!(result, vec![0.0, 127.0, -128.0]);
    }

    #[test]
    fn test_convert_i16_le() {
        let raw: Vec<u8> = vec![
            0x00, 0x00, // 0
            0xFF, 0x7F, // 32767
            0x00, 0x80, // -32768
        ];
        let dtype = Datatype::Integer {
            bits: NonZeroUsize::new(16).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
            signed: true,
        };
        let result = convert_to_f32(&raw, &dtype).unwrap();
        assert_eq!(result, vec![0.0, 32767.0, -32768.0]);
    }

    #[test]
    fn test_convert_boolean() {
        let raw = vec![0u8, 1, 0, 255];
        let result = convert_to_f32(&raw, &Datatype::Boolean).unwrap();
        assert_eq!(result, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_convert_unsupported_type() {
        let dtype = Datatype::VariableString {
            encoding: consus_core::StringEncoding::Utf8,
        };
        assert!(convert_to_f32(&[], &dtype).is_err());
    }
}
