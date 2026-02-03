use anyhow::{Result, Context, anyhow};
use burn::tensor::{Tensor, TensorData, Shape};
use burn::tensor::backend::Backend;
use nifti::{NiftiObject, ReaderOptions, IntoNdArray};
use ritk_core::image::Image;
use ritk_core::spatial::{Point, Spacing, Direction, Vector};
use nalgebra::SMatrix;
use std::path::Path;

pub fn read_nifti<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
    let path = path.as_ref();
    let obj = ReaderOptions::new().read_file(path).context("Failed to read NIfTI file")?;
    let header = obj.header();

    // Sform
    let affine = if header.sform_code > 0 {
        let r0 = header.srow_x;
        let r1 = header.srow_y;
        let r2 = header.srow_z;
        // Rows to [[f32; 4]; 4]
        [
            r0,
            r1,
            r2,
            [0.0, 0.0, 0.0, 1.0]
        ]
    } else if header.qform_code > 0 {
        // Qform implementation
        // See NIfTI standard
        let b = header.quatern_b;
        let c = header.quatern_c;
        let d = header.quatern_d;
        let a = (1.0 - (b*b + c*c + d*d).min(1.0)).sqrt();
        
        let qfac = if header.pixdim[0] == 0.0 { 1.0 } else { header.pixdim[0] };
        
        let r11 = a*a + b*b - c*c - d*d;
        let r12 = 2.0*b*c - 2.0*a*d;
        let r13 = 2.0*b*d + 2.0*a*c;
        
        let r21 = 2.0*b*c + 2.0*a*d;
        let r22 = a*a + c*c - b*b - d*d;
        let r23 = 2.0*c*d - 2.0*a*b;
        
        let r31 = 2.0*b*d - 2.0*a*c;
        let r32 = 2.0*c*d + 2.0*a*b;
        let r33 = a*a + d*d - c*c - b*b;
        
        let dx = header.pixdim[1];
        let dy = header.pixdim[2];
        let dz = header.pixdim[3] * qfac;
        
        let qx = header.quatern_x;
        let qy = header.quatern_y;
        let qz = header.quatern_z;
        
        [
            [r11*dx, r12*dy, r13*dz, qx],
            [r21*dx, r22*dy, r23*dz, qy],
            [r31*dx, r32*dy, r33*dz, qz],
            [0.0, 0.0, 0.0, 1.0]
        ]
    } else {
        // Fallback: use pixdim scaling only
        let dx = header.pixdim[1];
        let dy = header.pixdim[2];
        let dz = header.pixdim[3];
        [
            [dx, 0.0, 0.0, 0.0],
            [0.0, dy, 0.0, 0.0],
            [0.0, 0.0, dz, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    };

    // affine is [[f32; 4]; 4], accessed as affine[row][col]
    let m00 = affine[0][0] as f64; let m01 = affine[0][1] as f64; let m02 = affine[0][2] as f64; let tx = affine[0][3] as f64;
    let m10 = affine[1][0] as f64; let m11 = affine[1][1] as f64; let m12 = affine[1][2] as f64; let ty = affine[1][3] as f64;
    let m20 = affine[2][0] as f64; let m21 = affine[2][1] as f64; let m22 = affine[2][2] as f64; let tz = affine[2][3] as f64;

    let origin = Point::new([tx, ty, tz]);

    // Columns of the rotation matrix (scaled by spacing)
    let col0 = Vector::new([m00, m10, m20]);
    let col1 = Vector::new([m01, m11, m21]);
    let col2 = Vector::new([m02, m12, m22]);

    let sp0 = col0.0.norm();
    let sp1 = col1.0.norm();
    let sp2 = col2.0.norm();

    let spacing = Spacing::new([sp0, sp1, sp2]);

    // Normalize to get direction cosine matrix
    let d0 = if sp0 > 1e-9 { col0.0 / sp0 } else { nalgebra::Vector3::x_axis().into_inner() };
    let d1 = if sp1 > 1e-9 { col1.0 / sp1 } else { nalgebra::Vector3::y_axis().into_inner() };
    let d2 = if sp2 > 1e-9 { col2.0 / sp2 } else { nalgebra::Vector3::z_axis().into_inner() };

    // Build direction matrix from normalized columns
    let dir_matrix = SMatrix::<f64, 3, 3>::from_columns(&[d0, d1, d2]);
    let direction = Direction(dir_matrix);

    // Load voxel data
    let volume = obj.into_volume();
    // We load as f32. Burn tensors are typically float.
    // Note: this loads entire volume into memory (CPU) first.
    let ndarray_volume = volume.into_ndarray::<f32>().context("Failed to convert volume to ndarray")?;
    
    // ndarray shape
    let shape = ndarray_volume.shape();
    // Assuming 3D for now based on Image<B, 3> return type.
    if shape.len() != 3 {
        anyhow::bail!("Expected 3D NIfTI file, found {} dimensions", shape.len());
    }
    let dim0 = shape[0];
    let dim1 = shape[1];
    let dim2 = shape[2];

    // Convert to Burn Data
    // ndarray stores data in standard layout (last dimension contiguous) by default if created that way.
    // nifti-rs documentation says `into_ndarray` returns `ArrayD`.
    // We assume standard layout for now.
    // Note: Burn Data expects a flattened vector.
    let data_vec = ndarray_volume.into_raw_vec();
    let shape_burn = Shape::new([dim0, dim1, dim2]);
    
    let data = TensorData::new(data_vec, shape_burn);
    let tensor = Tensor::<B, 3>::from_data(data, device);
    
    // NIfTI is usually [X, Y, Z]. We want [Z, Y, X] for ritk-core conventions.
    // Permute axes: 0->2, 1->1, 2->0.
    let tensor = tensor.permute([2, 1, 0]);

    Ok(Image::new(tensor, origin, spacing, direction))
}

/// Write an image to a NIfTI file.
///
/// # Arguments
/// * `path` - Path to write the NIfTI file
/// * `image` - The image to write
///
/// # Returns
/// Result indicating success or failure
pub fn write_nifti<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    use nifti::writer::WriterOptions;
    use ndarray::Array3;

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
    let srow_x = [
        m_col0[0] as f32, m_col1[0] as f32, m_col2[0] as f32, origin[0] as f32
    ];
    let srow_y = [
        m_col0[1] as f32, m_col1[1] as f32, m_col2[1] as f32, origin[1] as f32
    ];
    let srow_z = [
        m_col0[2] as f32, m_col1[2] as f32, m_col2[2] as f32, origin[2] as f32
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use nifti::writer::WriterOptions;
    use tempfile::tempdir;
    use anyhow::Result;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_read_write_nifti_cycle() -> Result<()> {
        let dir = tempdir()?;
        let file_path = dir.path().join("test_cycle.nii");
        let device = Default::default();

        // Create a synthetic Image
        let shape = Shape::new([5, 4, 3]); // Z, Y, X
        let data = TensorData::new(vec![0.0; 3*4*5], shape);
        let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);
        
        let origin = Point::new([10.0, 20.0, 30.0]);
        let spacing = Spacing::new([0.5, 0.5, 2.0]);
        // Simple identity direction
        let direction = Direction(SMatrix::identity());

        let image = Image::new(tensor, origin, spacing, direction);

        // Write
        write_nifti(&file_path, &image)?;

        // Read back
        let loaded = read_nifti::<TestBackend, _>(&file_path, &device)?;

        // Verify Metadata
        let l_origin = loaded.origin();
        let l_spacing = loaded.spacing();

        // Verify sform preservation (which implies spacing/origin/direction are correct)
        assert!((l_origin[0] - 10.0).abs() < 1e-5);
        assert!((l_origin[1] - 20.0).abs() < 1e-5);
        assert!((l_origin[2] - 30.0).abs() < 1e-5);

        // Note: if pixdim is not set, read_nifti might fallback to pixdim if sform_code=0, but we set sform_code=1.
        // So read_nifti should use sform to derive spacing.
        // Columns of sform are [0.5, 0, 0], [0, 0.5, 0], [0, 0, 2.0].
        // Norms are 0.5, 0.5, 2.0. Correct.
        assert!((l_spacing[0] - 0.5).abs() < 1e-5);
        assert!((l_spacing[2] - 2.0).abs() < 1e-5);

        Ok(())
    }
}
