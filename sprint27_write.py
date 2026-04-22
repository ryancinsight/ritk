import pathlib

# ── resample.rs ──────────────────────────────────────────────────────────────
resample = '''//! Resample subcommand -- resamples a 3-D image to new voxel spacing.
//!
//! Physical extent E_d = N_d * S_d.
//! Output size: N_d_prime = max(1, round(E_d / S_d_prime)).
//! Identity transform (zero-offset TranslationTransform) maps output to input space.

use anyhow::{bail, Result};
use burn::tensor::backend::Backend as BurnBackend;
use burn::tensor::{Shape, Tensor, TensorData};
use clap;
use ritk_core::filter::resample::ResampleImageFilter;
use ritk_core::interpolation::linear::LinearInterpolator;
use ritk_core::interpolation::{BSplineInterpolator, Lanczos4Interpolator, NearestNeighborInterpolator};
use ritk_core::transform::translation::TranslationTransform;
use std::path::PathBuf;
use tracing::info;

use super::{read_image, write_image_inferred, Backend};

/// Resample an image to a new voxel spacing.
#[derive(clap::Args, Debug)]
pub struct ResampleArgs {
    #[arg(long)]
    pub input: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    /// New voxel spacing as "sz,sy,sx" (ZYX, comma-separated, positive floats).
    #[arg(long, default_value = "1.0,1.0,1.0")]
    pub spacing: String,
    /// Interpolation mode: nearest | linear | bspline | lanczos4
    #[arg(long, default_value = "linear")]
    pub interpolation: String,
}
'''

pathlib.Path('crates/ritk-cli/src/commands/resample.rs').write_text(resample, encoding='utf-8')
print('part-1 written', len(resample))

run_fn = '''
/// Execute the `resample` subcommand.
pub fn run(args: ResampleArgs) -> Result<()> {
    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        spacing = %args.spacing,
        interpolation = %args.interpolation,
        "resample: starting"
    );

    let parts: Vec<&str> = args.spacing.split(',').collect();
    if parts.len() != 3 {
        bail!("spacing must be sz,sy,sx (3 comma-separated values), got \'{}\'", args.spacing);
    }
    let new_sz: f64 = parts[0].trim().parse()
        .map_err(|_| anyhow::anyhow!("Invalid spacing value: \'{}\'", parts[0].trim()))?;
    let new_sy: f64 = parts[1].trim().parse()
        .map_err(|_| anyhow::anyhow!("Invalid spacing value: \'{}\'", parts[1].trim()))?;
    let new_sx: f64 = parts[2].trim().parse()
        .map_err(|_| anyhow::anyhow!("Invalid spacing value: \'{}\'", parts[2].trim()))?;
    if new_sz <= 0.0 || new_sy <= 0.0 || new_sx <= 0.0 {
        bail!("spacing values must be positive, got {},{},{}", new_sz, new_sy, new_sx);
    }

    let image = read_image(&args.input)?;
    let orig_dims = image.shape();
    let orig_spacing = *image.spacing();
    let orig_origin = *image.origin();
    let orig_dir = *image.direction();

    let new_nz = ((orig_dims[0] as f64 * orig_spacing[0]) / new_sz).round().max(1.0) as usize;
    let new_ny = ((orig_dims[1] as f64 * orig_spacing[1]) / new_sy).round().max(1.0) as usize;
    let new_nx = ((orig_dims[2] as f64 * orig_spacing[2]) / new_sx).round().max(1.0) as usize;

    use ritk_core::spatial::Spacing;
    let new_spacing = Spacing::new([new_sz, new_sy, new_sx]);
    let device: <Backend as BurnBackend>::Device = Default::default();
    let zero_t = Tensor::<Backend, 1>::from_data(
        TensorData::new(vec![0.0f32; 3], Shape::new([3])), &device,
    );

    let result = match args.interpolation.as_str() {
        "nearest" => ResampleImageFilter::new(
            [new_nz, new_ny, new_nx], orig_origin, new_spacing, orig_dir,
            TranslationTransform::<Backend, 3>::new(zero_t), NearestNeighborInterpolator::new(),
        ).apply(&image),
        "linear" => ResampleImageFilter::new(
            [new_nz, new_ny, new_nx], orig_origin, new_spacing, orig_dir,
            TranslationTransform::<Backend, 3>::new(zero_t), LinearInterpolator::new(),
        ).apply(&image),
        "bspline" => ResampleImageFilter::new(
            [new_nz, new_ny, new_nx], orig_origin, new_spacing, orig_dir,
            TranslationTransform::<Backend, 3>::new(zero_t), BSplineInterpolator::new(),
        ).apply(&image),
        "lanczos4" => ResampleImageFilter::new(
            [new_nz, new_ny, new_nx], orig_origin, new_spacing, orig_dir,
            TranslationTransform::<Backend, 3>::new(zero_t), Lanczos4Interpolator::new(),
        ).apply(&image),
        other => bail!(
            "Unknown interpolation mode \'{}\'..Accepted: nearest, linear, bspline, lanczos4",
            other
        ),
    };

    write_image_inferred(&args.output, &result)?;
    info!(new_size = format!("[{new_nz},{new_ny},{new_nx}]"), "resample: complete");
    Ok(())
}
'''

with open('crates/ritk-cli/src/commands/resample.rs', 'a', encoding='utf-8') as f:
    f.write(run_fn)
print('part-2 written')

tests_p1 = '''
#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    fn write_test_nifti(path: &std::path::Path, data: Vec<f32>, shape: [usize; 3], sp: [f64; 3]) {
        let dev: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(data, Shape::new(shape));
        let t = Tensor::<Backend, 3>::from_data(td, &dev);
        let img = Image::new(t, Point::new([0.0; 3]), Spacing::new(sp), Direction::identity());
        ritk_io::write_nifti::<Backend, _>(path, &img).expect("write ok");
    }

    #[test]
    fn test_resample_linear_same_spacing_preserves_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("in.nii");
        let output = dir.path().join("out.nii");
        write_test_nifti(&input, vec![1.0f32; 64], [4, 4, 4], [1.0, 1.0, 1.0]);
        let args = ResampleArgs {
            input: input.clone(), output: output.clone(),
            spacing: "1.0,1.0,1.0".to_string(), interpolation: "linear".to_string(),
        };
        run(args).expect("resample must succeed");
        assert!(output.exists());
        let dev: <Backend as BurnBackend>::Device = Default::default();
        let loaded = ritk_io::read_nifti::<Backend, _>(&output, &dev).unwrap();
        assert_eq!(loaded.shape(), [4, 4, 4], "shape preserved");
    }

    #[test]
    fn test_resample_half_spacing_doubles_size() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("in.nii");
        let output = dir.path().join("out.nii");
        write_test_nifti(&input, vec![1.0f32; 64], [4, 4, 4], [2.0, 2.0, 2.0]);
        let args = ResampleArgs {
            input: input.clone(), output: output.clone(),
            spacing: "1.0,1.0,1.0".to_string(), interpolation: "linear".to_string(),
        };
        run(args).unwrap();
        let dev: <Backend as BurnBackend>::Device = Default::default();
        let loaded = ritk_io::read_nifti::<Backend, _>(&output, &dev).unwrap();
        // E_d = 4 * 2.0 = 8; new size = round(8/1.0) = 8
        assert_eq!(loaded.shape(), [8, 8, 8], "halving spacing doubles voxel count");
    }
'''

with open('crates/ritk-cli/src/commands/resample.rs', 'a', encoding='utf-8') as f:
    f.write(tests_p1)
print('part-3 written')

tests_p2 = '''
    #[test]
    fn test_resample_nearest_constant_image() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("in.nii");
        let output = dir.path().join("out.nii");
        write_test_nifti(&input, vec![5.0f32; 27], [3, 3, 3], [1.0, 1.0, 1.0]);
        let args = ResampleArgs {
            input, output: output.clone(),
            spacing: "1.0,1.0,1.0".to_string(), interpolation: "nearest".to_string(),
        };
        run(args).unwrap();
        let dev: <Backend as BurnBackend>::Device = Default::default();
        let loaded = ritk_io::read_nifti::<Backend, _>(&output, &dev).unwrap();
        let vals: Vec<f32> = loaded.data().clone().into_data()
            .as_slice::<f32>().unwrap().to_vec();
        for &v in &vals {
            assert!((v - 5.0).abs() < 1e-3, "constant image must stay constant, got {v}");
        }
    }

    #[test]
    fn test_resample_unknown_mode_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("in.nii");
        let output = dir.path().join("out.nii");
        write_test_nifti(&input, vec![0.0f32; 8], [2, 2, 2], [1.0, 1.0, 1.0]);
        let args = ResampleArgs {
            input, output,
            spacing: "1.0,1.0,1.0".to_string(), interpolation: "cubic".to_string(),
        };
        let result = run(args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown interpolation mode"));
    }

    #[test]
    fn test_resample_invalid_spacing_string_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("in.nii");
        let output = dir.path().join("out.nii");
        write_test_nifti(&input, vec![0.0f32; 8], [2, 2, 2], [1.0, 1.0, 1.0]);
        let args = ResampleArgs {
            input, output,
            spacing: "1.0,2.0".to_string(), interpolation: "linear".to_string(),
        };
        let result = run(args);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("3"), "error must mention 3 values: {msg}");
    }
}
'''

with open('crates/ritk-cli/src/commands/resample.rs', 'a', encoding='utf-8') as f:
    f.write(tests_p2)
print('part-4 written')

# ── ritk-python/src/filter.rs: add imports ────────────────────────────────────
import re

py_filter = pathlib.Path('crates/ritk-python/src/filter.rs')
src = py_filter.read_text(encoding='utf-8')

new_imports = '''use ritk_core::filter::{LabelErosion, LabelOpening, LabelClosing, MorphologicalReconstruction, ReconstructionMode, ResampleImageFilter};
use ritk_core::interpolation::linear::LinearInterpolator;
use ritk_core::interpolation::{BSplineInterpolator, Lanczos4Interpolator, NearestNeighborInterpolator};
use ritk_core::transform::translation::TranslationTransform;
use ritk_core::spatial::Spacing as CoreSpacing;
use burn::tensor::{Shape, Tensor, TensorData};
use burn::tensor::backend::Backend as BurnBackend;
'''

# Insert new imports after the last existing use block
insert_after = 'use ritk_core::filter::{\n    BinaryThresholdImageFilter, IntensityWindowingFilter, RescaleIntensityFilter,\n    SigmoidImageFilter, ThresholdImageFilter,\n};'
src2 = src.replace(insert_after, insert_after + '\n' + new_imports, 1)
if src2 == src:
    print('ERROR: import anchor not found')
else:
    py_filter.write_text(src2, encoding='utf-8')
    print('py filter.rs imports added')

# ── 5 new pyfunction definitions (before register) ────────────────────────────
new_pyfns = '''
// ── label_erosion ──────────────────────────────────────────────────────────────

/// Erode labeled regions in a 3-D label volume.
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_erosion(py: Python<\'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        LabelErosion::new(radius).apply(img.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── label_opening ─────────────────────────────────────────────────────────────

/// Opening on a 3-D label volume (erosion followed by dilation).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_opening(py: Python<\'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        LabelOpening::new(radius).apply(img.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── label_closing ─────────────────────────────────────────────────────────────

/// Closing on a 3-D label volume (dilation followed by erosion).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_closing(py: Python<\'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        LabelClosing::new(radius).apply(img.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── morphological_reconstruction ─────────────────────────────────────────────

/// Geodesic morphological reconstruction.
#[pyfunction]
#[pyo3(signature = (marker, mask, mode = "dilation"))]
pub fn morphological_reconstruction(
    py: Python<\'_>,
    marker: &PyImage,
    mask: &PyImage,
    mode: &str,
) -> PyResult<PyImage> {
    let recon_mode = match mode {
        "dilation" => ReconstructionMode::Dilation,
        "erosion"  => ReconstructionMode::Erosion,
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown reconstruction mode \'{}\'. Use \'dilation\' or \'erosion\'.", other
        ))),
    };
    let marker_arc = std::sync::Arc::clone(&marker.inner);
    let mask_arc   = std::sync::Arc::clone(&mask.inner);
    let result = py.allow_threads(|| {
        MorphologicalReconstruction::new(recon_mode)
            .apply(marker_arc.as_ref(), mask_arc.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── resample_image ────────────────────────────────────────────────────────────

/// Resample a 3-D image to new voxel spacing.
///
/// Output size N_d_prime = max(1, round(N_d * S_d / S_d_prime)).
#[pyfunction]
#[pyo3(signature = (image, spacing_z=1.0_f64, spacing_y=1.0_f64, spacing_x=1.0_f64, mode="linear"))]
pub fn resample_image(
    py: Python<\'_>,
    image: &PyImage,
    spacing_z: f64,
    spacing_y: f64,
    spacing_x: f64,
    mode: &str,
) -> PyResult<PyImage> {
    if spacing_z <= 0.0 || spacing_y <= 0.0 || spacing_x <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "spacing values must be positive, got ({spacing_z},{spacing_y},{spacing_x})"
        )));
    }
    let mode = mode.to_string();
    let inner = std::sync::Arc::clone(&image.inner);

    let result = py.allow_threads(move || -> Result<_, String> {
        let orig_dims = inner.shape();
        let orig_sp   = *inner.spacing();
        let orig_orig = *inner.origin();
        let orig_dir  = *inner.direction();

        let new_nz = ((orig_dims[0] as f64 * orig_sp[0]) / spacing_z).round().max(1.0) as usize;
        let new_ny = ((orig_dims[1] as f64 * orig_sp[1]) / spacing_y).round().max(1.0) as usize;
        let new_nx = ((orig_dims[2] as f64 * orig_sp[2]) / spacing_x).round().max(1.0) as usize;

        let new_sp = CoreSpacing::new([spacing_z, spacing_y, spacing_x]);
        let device: <Backend as BurnBackend>::Device = Default::default();
        let zero_t = Tensor::<Backend, 1>::from_data(
            TensorData::new(vec![0.0f32; 3], Shape::new([3])), &device
        );

        match mode.as_str() {
            "nearest" => Ok(ResampleImageFilter::new(
                [new_nz, new_ny, new_nx], orig_orig, new_sp, orig_dir,
                TranslationTransform::<Backend, 3>::new(zero_t),
                NearestNeighborInterpolator::new(),
            ).apply(inner.as_ref())),
            "linear" => Ok(ResampleImageFilter::new(
                [new_nz, new_ny, new_nx], orig_orig, new_sp, orig_dir,
                TranslationTransform::<Backend, 3>::new(zero_t),
                LinearInterpolator::new(),
            ).apply(inner.as_ref())),
            "bspline" => Ok(ResampleImageFilter::new(
                [new_nz, new_ny, new_nx], orig_orig, new_sp, orig_dir,
                TranslationTransform::<Backend, 3>::new(zero_t),
                BSplineInterpolator::new(),
            ).apply(inner.as_ref())),
            "lanczos4" => Ok(ResampleImageFilter::new(
                [new_nz, new_ny, new_nx], orig_orig, new_sp, orig_dir,
                TranslationTransform::<Backend, 3>::new(zero_t),
                Lanczos4Interpolator::new(),
            ).apply(inner.as_ref())),
            other => Err(format!(
                "Unknown interpolation mode \'{}\'. Use: nearest, linear, bspline, lanczos4", other
            )),
        }
    }).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    Ok(into_py_image(result))
}
'''

src3 = py_filter.read_text(encoding='utf-8')
# Insert before pub fn register
anchor = '/// Register the `filter` submodule.'
src4 = src3.replace(anchor, new_pyfns + '\n' + anchor, 1)
if src4 == src3:
    print('ERROR: register anchor not found')
else:
    py_filter.write_text(src4, encoding='utf-8')
    print('py filter.rs: 5 pyfunctions added')

# ── Add 5 add_function calls in register() before parent.add_submodule ────────
src5 = py_filter.read_text(encoding='utf-8')
old_sub = '    parent.add_submodule(&m)?;'
new_add = '''    m.add_function(wrap_pyfunction!(label_erosion, &m)?)?;
    m.add_function(wrap_pyfunction!(label_opening, &m)?)?;
    m.add_function(wrap_pyfunction!(label_closing, &m)?)?;
    m.add_function(wrap_pyfunction!(morphological_reconstruction, &m)?)?;
    m.add_function(wrap_pyfunction!(resample_image, &m)?)?;
    parent.add_submodule(&m)?;'''
src6 = src5.replace(old_sub, new_add, 1)
if src6 == src5:
    print('ERROR: add_submodule anchor not found')
else:
    py_filter.write_text(src6, encoding='utf-8')
    print('py filter.rs: 5 register calls added')
