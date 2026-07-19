use super::*;
use ritk_registration::classical::{image_to_leto_volume, leto_volume_to_image};

/// Run rigid-mi or affine-mi registration via the classical MI engine.
pub(super) fn run_mi_registration(args: &RegisterArgs) -> Result<()> {
    let fixed_native = super::super::read_image(&args.fixed)?;
    let moving_native = super::super::read_image(&args.moving)?;

    let filter = GaussianFilter::<crate::commands::Backend>::new(vec![args.sigma_fixed; 3]);
    let backend = crate::commands::Backend::default();
    let fixed_smoothed = filter.apply_native(&fixed_native, &backend)?;
    let moving_smoothed = filter.apply_native(&moving_native, &backend)?;

    let fixed_volume = image_to_leto_volume(&fixed_smoothed)?;
    let moving_volume = image_to_leto_volume(&moving_smoothed)?;
    let config = ClassicalConfig {
        max_iterations: args.iterations,
        tolerance: 1e-6,
        step_multiplier: 1.0,
    };
    let registration = ImageRegistration::with_config(config, MutualInformationMetric::default());
    let initial_transform = ritk_registration::AffineTransform::IDENTITY;
    let result = match &args.method {
        RegistrationMethod::RigidMi => registration
            .rigid_registration_mutual_info(&moving_volume, &fixed_volume, &initial_transform)
            .with_context(|| "rigid MI registration failed")?,
        RegistrationMethod::AffineMi => registration
            .affine_registration_mutual_info(&moving_volume, &fixed_volume, &initial_transform)
            .with_context(|| "affine MI registration failed")?,
        _ => unreachable!("run_mi_registration called with non-MI method"),
    };

    let moving_original = super::super::read_image(&args.moving)?;
    let moving_original_volume = image_to_leto_volume(&moving_original)?;
    let warped_volume = spatial::apply_transform(&moving_original_volume, &result.transform);
    let warped_native = leto_volume_to_image(&warped_volume, &fixed_native, &backend)?;
    let format = super::super::infer_format(&args.output)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer output format: {}", args.output.display()))?;
    super::super::write_image(&args.output, &warped_native, format)?;

    if let Some(transform_path) = &args.output_transform {
        let json = serde_json::to_string_pretty(result.transform.as_array())
            .context("Failed to serialise transform to JSON")?;
        std::fs::write(transform_path, &json).with_context(|| {
            format!("Failed to write transform to {}", transform_path.display())
        })?;
        info!(
            "register: transform written path={}",
            transform_path.display()
        );
    }

    let quality = &result.quality;
    println!(
        "Registered {} → {} (method={}, iterations={}, converged={:?}, MI={:.6}, cost={:.6})",
        args.moving.display(),
        args.output.display(),
        args.method,
        quality.iterations,
        quality.convergence,
        quality.mutual_information,
        quality.final_cost,
    );
    info!(
        "register: MI registration complete method={} iterations={} converged={:?} mi={} final_cost={}",
        args.method,
        quality.iterations,
        quality.convergence,
        quality.mutual_information,
        quality.final_cost
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use ritk_image::Image;
    use ritk_io::{
        format::nifti::native::{NiftiReader, NiftiWriter},
        ImageReader, ImageWriter,
    };
    use ritk_registration::demons::DemonsVariant;
    use ritk_spatial::{Direction, Point, Spacing};
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    fn make_ramp_image() -> Image<f32, SequentialBackend, 3> {
        let values = (0..64)
            .scan(0.0, |next, _| {
                let value = *next;
                *next += 4.0;
                Some(value)
            })
            .collect();
        Image::from_flat_on(
            values,
            [4, 4, 4],
            Point::origin(),
            Spacing::uniform(1.0),
            Direction::identity(),
            &SequentialBackend,
        )
        .expect("ramp image has matching shape and storage")
    }

    fn write_fixture(path: &Path, image: &Image<f32, SequentialBackend, 3>) {
        ImageWriter::write(&NiftiWriter::new(SequentialBackend), path, image)
            .expect("native NIfTI fixture write");
    }

    fn read_fixture(path: &Path) -> Image<f32, SequentialBackend, 3> {
        ImageReader::read(&NiftiReader::new(SequentialBackend), path)
            .expect("native NIfTI fixture read")
    }

    fn registration_args(
        fixed: PathBuf,
        moving: PathBuf,
        output: PathBuf,
        method: RegistrationMethod,
    ) -> RegisterArgs {
        RegisterArgs {
            fixed,
            moving,
            output,
            method,
            output_transform: None,
            iterations: 3,
            sigma_fixed: GaussianSigma::default(),
            levels: 3,
            variant: DemonsVariant::Classic,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: CliInverseConsistency::Relaxed,
            num_time_steps: 2,
            kernel_sigma: GaussianSigma::new_unchecked(3.0),
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        }
    }

    #[test]
    fn rigid_mi_preserves_identical_native_fixture() {
        let directory = tempdir().expect("temporary directory");
        let fixed_path = directory.path().join("fixed.nii");
        let moving_path = directory.path().join("moving.nii");
        let output_path = directory.path().join("warped.nii");
        let image = make_ramp_image();
        write_fixture(&fixed_path, &image);
        write_fixture(&moving_path, &image);

        run(registration_args(
            fixed_path,
            moving_path,
            output_path.clone(),
            RegistrationMethod::RigidMi,
        ))
        .expect("rigid MI registration");

        let warped = read_fixture(&output_path);
        assert_eq!(warped.shape(), image.shape());
        assert_eq!(warped.data_slice().unwrap(), image.data_slice().unwrap());
    }

    #[test]
    fn affine_mi_preserves_identical_native_fixture() {
        let directory = tempdir().expect("temporary directory");
        let fixed_path = directory.path().join("fixed.nii");
        let moving_path = directory.path().join("moving.nii");
        let output_path = directory.path().join("warped.nii");
        let image = make_ramp_image();
        write_fixture(&fixed_path, &image);
        write_fixture(&moving_path, &image);

        run(registration_args(
            fixed_path,
            moving_path,
            output_path.clone(),
            RegistrationMethod::AffineMi,
        ))
        .expect("affine MI registration");

        let warped = read_fixture(&output_path);
        assert_eq!(warped.shape(), image.shape());
        assert_eq!(warped.data_slice().unwrap(), image.data_slice().unwrap());
    }

    #[test]
    fn rigid_mi_writes_identity_transform_for_identical_fixture() {
        let directory = tempdir().expect("temporary directory");
        let fixed_path = directory.path().join("fixed.nii");
        let moving_path = directory.path().join("moving.nii");
        let output_path = directory.path().join("warped.nii");
        let transform_path = directory.path().join("transform.json");
        let image = make_ramp_image();
        write_fixture(&fixed_path, &image);
        write_fixture(&moving_path, &image);

        let mut args = registration_args(
            fixed_path,
            moving_path,
            output_path,
            RegistrationMethod::RigidMi,
        );
        args.output_transform = Some(transform_path.clone());
        run(args).expect("rigid MI registration with transform output");

        let json = std::fs::read_to_string(transform_path).expect("transform JSON output");
        let transform: Vec<f64> =
            serde_json::from_str(&json).expect("transform file must be valid JSON");
        assert_eq!(
            transform,
            ritk_registration::AffineTransform::IDENTITY
                .as_array()
                .to_vec()
        );
    }
}
