use anyhow::Result;
use burn::tensor::backend::Backend;
use nifti::NiftiHeader;
use ritk_core::image::Image;
use std::path::Path;

/// Write an image to a NIfTI file with full sform spatial metadata.
///
/// # Spatial convention
/// RITK tensors are ordered [Z, Y, X]. The sform affine maps voxel [i,j,k] to physical [x,y,z]:
///   srow_x = [M_col0[0], M_col1[0], M_col2[0], origin[0]]
///   srow_y = [M_col0[1], M_col1[1], M_col2[1], origin[1]]
///   srow_z = [M_col0[2], M_col1[2], M_col2[2], origin[2]]
/// where M_colJ = direction.column(J) * spacing[J].
pub fn write_nifti<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    use ndarray::Array3;
    use nifti::writer::WriterOptions;

    let tensor = image.data().clone().permute([2, 1, 0]);
    let data = tensor.to_data();
    let slice = match data.as_slice::<f32>() {
        Ok(s) => s,
        Err(e) => return Err(anyhow::anyhow!("Failed to get tensor data: {:?}", e)),
    };

    let shape = image.shape();
    let nx = shape[2];
    let ny = shape[1];
    let nz = shape[0];

    let array = Array3::from_shape_vec((nx, ny, nz), slice.to_vec())
        .map_err(|e| anyhow::anyhow!("Failed to create ndarray: {}", e))?;

    let path_ref = path.as_ref();

    let direction = image.direction().0;
    let origin = image.origin();
    let spacing = image.spacing();

    let m_col0 = direction.column(0) * spacing[0];
    let m_col1 = direction.column(1) * spacing[1];
    let m_col2 = direction.column(2) * spacing[2];

    let srow_x = [m_col0[0] as f32, m_col1[0] as f32, m_col2[0] as f32, origin[0] as f32];
    let srow_y = [m_col0[1] as f32, m_col1[1] as f32, m_col2[1] as f32, origin[1] as f32];
    let srow_z = [m_col0[2] as f32, m_col1[2] as f32, m_col2[2] as f32, origin[2] as f32];

    let header = NiftiHeader {
        sform_code: 1,
        qform_code: 0,
        srow_x,
        srow_y,
        srow_z,
        pixdim: [1.0, spacing[0] as f32, spacing[1] as f32, spacing[2] as f32,
                 1.0, 1.0, 1.0, 1.0],
        xyzt_units: 2, // NIFTI_UNITS_MM
        ..NiftiHeader::default()
    };

    WriterOptions::new(path_ref)
        .reference_header(&header)
        .write_nifti(&array)
        .map_err(|e| anyhow::anyhow!("Failed to write NIfTI file: {}", e))?;

    Ok(())
}
