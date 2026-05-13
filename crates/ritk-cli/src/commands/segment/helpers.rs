use anyhow::{anyhow, Context, Result};

use super::super::Backend;

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
    let td = image.data().clone().into_data();
    let slice = td
        .as_slice::<f32>()
        .expect("segmentation output must contain f32 data");
    slice.iter().filter(|&&v| v > 0.5).count()
}
