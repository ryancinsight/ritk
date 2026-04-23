//! Resample subcommand -- resamples a 3-D image to new voxel spacing.
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

/// Execute the `resample` subcommand.
pub fn run(args: ResampleArgs) -> Result<()> {
    info!("resample: starting input={} output={} spacing={} interpolation={}", args.input.display(), args.output.display(), args.spacing, args.interpolation);

    let parts: Vec<&str> = args.spacing.split(',').collect();
    if parts.len() != 3 {
        bail!("spacing must be sz,sy,sx (3 comma-separated values), got '{}'", args.spacing);
    }
    let new_sz: f64 = parts[0].trim().parse()
        .map_err(|_| anyhow::anyhow!("Invalid spacing value: '{}'", parts[0].trim()))?;
    let new_sy: f64 = parts[1].trim().parse()
        .map_err(|_| anyhow::anyhow!("Invalid spacing value: '{}'", parts[1].trim()))?;
    let new_sx: f64 = parts[2].trim().parse()
        .map_err(|_| anyhow::anyhow!("Invalid spacing value: '{}'", parts[2].trim()))?;
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
            "Unknown interpolation mode '{}'..Accepted: nearest, linear, bspline, lanczos4",
            other
        ),
    };

    write_image_inferred(&args.output, &result)?;
    info!(new_size = format!("[{new_nz},{new_ny},{new_nx}]"), "resample: complete");
    Ok(())
}

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
        // Uses NRRD: NIfTI writer does not persist sform/pixdim, so spacing is lost on
        // NIfTI round-trip.  NRRD preserves all spatial metadata.
        //
        // Physical extent: E_d = 4 * 2.0 = 8.0
        // New size: N_d = round(E_d / 1.0) = 8
        let dir = tempdir().unwrap();
        let input = dir.path().join("in.nrrd");
        let output = dir.path().join("out.nrrd");
        let dev: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![1.0f32; 64], Shape::new([4, 4, 4]));
        let t = Tensor::<Backend, 3>::from_data(td, &dev);
        let img = Image::new(
            t,
            Point::new([0.0; 3]),
            Spacing::new([2.0, 2.0, 2.0]),
            Direction::identity(),
        );
        ritk_io::write_nrrd::<Backend, _>(&input, &img).expect("write_nrrd must succeed");
        let args = ResampleArgs {
            input: input.clone(), output: output.clone(),
            spacing: "1.0,1.0,1.0".to_string(), interpolation: "linear".to_string(),
        };
        run(args).unwrap();
        let loaded = ritk_io::read_nrrd::<Backend, _>(&output, &dev).unwrap();
        assert_eq!(loaded.shape(), [8, 8, 8], "halving spacing must double voxel count");
    }

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
