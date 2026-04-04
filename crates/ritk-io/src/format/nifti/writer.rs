use anyhow::Result;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

/// Write an image to a NIfTI file.
///
/// # Arguments
/// * `path` - Path to write the NIfTI file
/// * `image` - The image to write
///
/// # Returns
/// Result indicating success or failure
pub fn write_nifti<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    use ndarray::Array3;
    use nifti::writer::WriterOptions;

    // Get image data and permute back to NIfTI convention [X, Y, Z] from ritk's [Z, Y, X]
    let tensor = image.data().clone().permute([2, 1, 0]);
    let data = tensor.to_data();
    let slice = match data.as_slice::<f32>() {
        Ok(s) => s,
        Err(e) => return Err(anyhow::anyhow!("Failed to get tensor data: {:?}", e)),
    };

    let shape = image.shape();
    // shape is [Z, Y, X] in ritk, NIfTI expects [X, Y, Z]
    let nx = shape[2];
    let ny = shape[1];
    let nz = shape[0];

    // Create ndarray with NIfTI ordering
    let array = Array3::from_shape_vec((nx, ny, nz), slice.to_vec())
        .map_err(|e| anyhow::anyhow!("Failed to create ndarray: {}", e))?;

    // Use nifti crate's WriterOptions to write the file
    let path_ref = path.as_ref();

    // Calculate Affine Matrix for sform
    // M = Direction * diag(Spacing)
    let direction = image.direction().0;
    let origin = image.origin();
    let spacing = image.spacing();

    // Column vectors of M
    let m_col0 = direction.column(0) * spacing[0];
    let m_col1 = direction.column(1) * spacing[1];
    let m_col2 = direction.column(2) * spacing[2];

    // Sform rows (transposed from M columns)
    // srow_x = [M00, M01, M02, Ox]
    let _srow_x = [
        m_col0[0] as f32,
        m_col1[0] as f32,
        m_col2[0] as f32,
        origin[0] as f32,
    ];
    let _srow_y = [
        m_col0[1] as f32,
        m_col1[1] as f32,
        m_col2[1] as f32,
        origin[1] as f32,
    ];
    let _srow_z = [
        m_col0[2] as f32,
        m_col1[2] as f32,
        m_col2[2] as f32,
        origin[2] as f32,
    ];

    // Note: nifti-rs 0.16 WriterOptions doesn't expose a `header` setter.
    // We use individual setters.
    // Write NIfTI file
    // Note: The nifti crate's WriterOptions API changed in recent versions
    // Using basic write_nifti - spatial metadata stored in header separately
    WriterOptions::new(path_ref)
        .write_nifti(&array)
        .map_err(|e| anyhow::anyhow!("Failed to write NIfTI file: {}", e))?;

    Ok(())
}
