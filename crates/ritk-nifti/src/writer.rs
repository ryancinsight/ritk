use anyhow::Result;
use burn::tensor::backend::Backend;
use nifti::NiftiHeader;
use ritk_core::image::Image;
use std::path::Path;

/// Write a label map to a NIfTI-1 file with `DT_UINT32` data type.
///
/// # Spatial convention
///
/// `shape` is `[nz, ny, nx]` (ZYX).  `spacing` is `[dz, dy, dx]`.
/// `direction` is the 3×3 direction-cosine matrix in **row-major flat layout**,
/// matching `LoadedVolume::direction`.  The sform affine follows the same
/// convention as [`write_nifti`]:
///
/// ```text
/// srow_x[j] = direction_col_j[0] * spacing[j]   (j=0,1,2)
/// srow_y[j] = direction_col_j[1] * spacing[j]
/// srow_z[j] = direction_col_j[2] * spacing[j]
/// srow_?[3]  = origin[?]
/// ```
///
/// # Errors
///
/// Returns `Err` when `labels.len() != nz * ny * nx` or when the nifti
/// writer reports a failure.
pub fn write_nifti_labels<P: AsRef<Path>>(
    path: P,
    labels: &[u32],
    shape: [usize; 3], // [nz, ny, nx]
    origin: [f32; 3],
    spacing: [f32; 3], // [dz, dy, dx]
    direction: [f32; 9], // row-major 3×3
) -> Result<()> {
    use ndarray::Array3;
    use nifti::writer::WriterOptions;

    let [nz, ny, nx] = shape;
    let expected = nz * ny * nx;
    if labels.len() != expected {
        anyhow::bail!(
            "write_nifti_labels: labels.len()={} != shape product {}",
            labels.len(),
            expected
        );
    }

    // Fill the ndarray using logical indexing so the writer is independent of
    // the in-memory layout chosen by ndarray.  The NIfTI crate treats dim[1]=x,
    // dim[2]=y, dim[3]=z, so the array shape must be (nx, ny, nz) with
    // array[[x, y, z]] = label at RITK ZYX position (z, y, x).
    let mut array = Array3::<u32>::zeros((nx, ny, nz));
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                array[[x, y, z]] = labels[z * ny * nx + y * nx + x];
            }
        }
    }

    let [dz, dy, dx] = spacing;
    let d = &direction;
    // column j of row-major direction (d[row*3+col]): col(j) = [d[j], d[3+j], d[6+j]]
    let srow_x = [d[0] * dz, d[1] * dy, d[2] * dx, origin[0]];
    let srow_y = [d[3] * dz, d[4] * dy, d[5] * dx, origin[1]];
    let srow_z = [d[6] * dz, d[7] * dy, d[8] * dx, origin[2]];

    let header = NiftiHeader {
        sform_code: 1,
        qform_code: 0,
        srow_x,
        srow_y,
        srow_z,
        pixdim: [1.0, dz, dy, dx, 1.0, 1.0, 1.0, 1.0],
        xyzt_units: 2, // NIFTI_UNITS_MM
        ..NiftiHeader::default()
    };

    WriterOptions::new(path.as_ref())
        .reference_header(&header)
        .write_nifti(&array)
        .map_err(|e| anyhow::anyhow!("Failed to write NIfTI labels: {e}"))?;

    Ok(())
}

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

    let srow_x = [
        m_col0[0] as f32,
        m_col1[0] as f32,
        m_col2[0] as f32,
        origin[0] as f32,
    ];
    let srow_y = [
        m_col0[1] as f32,
        m_col1[1] as f32,
        m_col2[1] as f32,
        origin[1] as f32,
    ];
    let srow_z = [
        m_col0[2] as f32,
        m_col1[2] as f32,
        m_col2[2] as f32,
        origin[2] as f32,
    ];

    let header = NiftiHeader {
        sform_code: 1,
        qform_code: 0,
        srow_x,
        srow_y,
        srow_z,
        pixdim: [
            1.0,
            spacing[0] as f32,
            spacing[1] as f32,
            spacing[2] as f32,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        xyzt_units: 2, // NIFTI_UNITS_MM
        ..NiftiHeader::default()
    };

    WriterOptions::new(path_ref)
        .reference_header(&header)
        .write_nifti(&array)
        .map_err(|e| anyhow::anyhow!("Failed to write NIfTI file: {}", e))?;

    Ok(())
}
