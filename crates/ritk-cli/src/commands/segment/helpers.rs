use anyhow::{anyhow, Context, Result};

use super::super::{
    infer_format, is_native_read_capable, is_native_write_capable, read_image_native, Backend,
    NativeBackend,
};

pub(crate) fn read_native_input(
    input: &std::path::Path,
    output: &std::path::Path,
    operation: &str,
) -> Result<(
    ritk_image::native::Image<f32, NativeBackend, 3>,
    ritk_io::ImageFormat,
)> {
    let input_format = infer_format(input)
        .ok_or_else(|| anyhow!("Cannot infer input format: {}", input.display()))?;
    let output_format = infer_format(output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", output.display()))?;
    anyhow::ensure!(
        is_native_read_capable(input_format) && is_native_write_capable(output_format),
        "{operation} requires native input/output formats"
    );
    Ok((read_image_native(input)?, output_format))
}

/// Parse a `"Z,Y,X"` string into a `[usize; 3]` seed voxel index.
///
/// # Errors
/// Returns an error when the string does not contain exactly three
/// comma-separated non-negative integer tokens.
pub(crate) fn parse_seed(s: &str) -> Result<[usize; 3]> {
    let parts: Vec<&str> = s.splitn(4, ',').collect();
    if parts.len() != 3 {
        return Err(anyhow!(
            "Seed must be provided as Z,Y,X (three comma-separated integers), got: '{s}'"
        ));
    }
    let z = parts[0]
        .trim()
        .parse::<usize>()
        .with_context(|| format!("Invalid Z component '{}' in seed '{s}'", parts[0]))?;
    let y = parts[1]
        .trim()
        .parse::<usize>()
        .with_context(|| format!("Invalid Y component '{}' in seed '{s}'", parts[1]))?;
    let x = parts[2]
        .trim()
        .parse::<usize>()
        .with_context(|| format!("Invalid X component '{}' in seed '{s}'", parts[2]))?;
    Ok([z, y, x])
}

/// Count the number of voxels with value > 0.5 in `image`.
///
/// Suitable for binary (0.0 / 1.0) masks produced by Otsu and
/// connected-threshold segmentation.
///
/// # Panics
/// Panics if the tensor data cannot be extracted as `f32`.
pub(crate) fn count_foreground(image: &ritk_core::image::Image<Backend, 3>) -> usize {
    image.with_data_slice(|slice| slice.iter().filter(|&&v| v > 0.5).count())
}

pub(crate) fn count_native_foreground(
    image: &ritk_image::native::Image<f32, NativeBackend, 3>,
) -> Result<usize> {
    Ok(image
        .data_slice()?
        .iter()
        .filter(|&&value| value > 0.5)
        .count())
}
