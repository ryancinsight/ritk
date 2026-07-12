use anyhow::{anyhow, Result};
use tracing::info;

#[cfg(test)]
use super::Backend;
use super::{
    super::{
        infer_format, is_native_read_capable, is_native_write_capable, read_image_native,
        write_image_native, NativeBackend,
    },
    read_image, write_image_inferred, FilterArgs,
};

pub(super) fn run_bed_separation(args: &FilterArgs) -> Result<()> {
    use ritk_filter::{BedSeparationConfig, BedSeparationFilter};

    let image = read_image(&args.input)?;

    let config = BedSeparationConfig {
        body_threshold: args.bed.body_threshold,
        closing_radius: args.bed.closing_radius,
        opening_radius: args.bed.opening_radius,
        outside_value: args.bed.outside_value,
        ..Default::default()
    };
    let filtered = BedSeparationFilter::new(config).apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied bed-separation (body_threshold={}, closing_radius={}, opening_radius={}, outside={}) to {} -> {}",
        args.bed.body_threshold,
        args.bed.closing_radius,
        args.bed.opening_radius,
        args.bed.outside_value,
        args.input.display(),
        args.output.display()
    );
    info!("filter: bed-separation complete");

    Ok(())
}

pub(super) fn run_rescale_intensity(args: &FilterArgs) -> Result<()> {
    use ritk_filter::RescaleIntensityFilter;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_native_read_capable(input_format),
        "rescale-intensity requires native input support for {:?}",
        input_format
    );
    anyhow::ensure!(
        is_native_write_capable(output_format),
        "rescale-intensity requires native output support for {:?}",
        output_format
    );
    let image = read_image_native(&args.input)?;
    let backend = NativeBackend::default();
    let filtered = RescaleIntensityFilter::new(args.range.out_min, args.range.out_max)
        .apply_native(&image, &backend)?;

    write_image_native(&args.output, &filtered, output_format)?;

    println!(
        "Applied rescale-intensity (out=[{},{}]) to {} -> {}",
        args.range.out_min,
        args.range.out_max,
        args.input.display(),
        args.output.display()
    );
    info!("filter: rescale-intensity complete");

    Ok(())
}

pub(super) fn run_intensity_windowing(args: &FilterArgs) -> Result<()> {
    use ritk_filter::IntensityWindowingFilter;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_native_read_capable(input_format),
        "intensity-windowing requires native input support for {:?}",
        input_format
    );
    anyhow::ensure!(
        is_native_write_capable(output_format),
        "intensity-windowing requires native output support for {:?}",
        output_format
    );
    let image = read_image_native(&args.input)?;
    let backend = NativeBackend::default();
    let filtered = IntensityWindowingFilter::new(
        args.window.window_min,
        args.window.window_max,
        args.range.out_min,
        args.range.out_max,
    )
    .apply_native(&image, &backend)?;

    write_image_native(&args.output, &filtered, output_format)?;

    println!(
        "Applied intensity-windowing (window=[{},{}], out=[{},{}]) to {} -> {}",
        args.window.window_min,
        args.window.window_max,
        args.range.out_min,
        args.range.out_max,
        args.input.display(),
        args.output.display()
    );
    info!("filter: intensity-windowing complete");

    Ok(())
}

pub(super) fn run_threshold_below(args: &FilterArgs) -> Result<()> {
    use ritk_filter::ThresholdImageFilter;

    let image = read_image(&args.input)?;
    let filtered =
        ThresholdImageFilter::below(args.threshold.threshold_value, args.band.outside_value)
            .apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied threshold-below (threshold={}, outside={}) to {} -> {}",
        args.threshold.threshold_value,
        args.band.outside_value,
        args.input.display(),
        args.output.display()
    );
    info!("filter: threshold-below complete");

    Ok(())
}

pub(super) fn run_threshold_above(args: &FilterArgs) -> Result<()> {
    use ritk_filter::ThresholdImageFilter;

    let image = read_image(&args.input)?;
    let filtered =
        ThresholdImageFilter::above(args.threshold.threshold_value, args.band.outside_value)
            .apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied threshold-above (threshold={}, outside={}) to {} -> {}",
        args.threshold.threshold_value,
        args.band.outside_value,
        args.input.display(),
        args.output.display()
    );
    info!("filter: threshold-above complete");

    Ok(())
}

pub(super) fn run_threshold_outside(args: &FilterArgs) -> Result<()> {
    use ritk_filter::ThresholdImageFilter;

    let image = read_image(&args.input)?;
    let filtered = ThresholdImageFilter::outside(
        args.band.lower_threshold,
        args.band.upper_threshold,
        args.band.outside_value,
    )
    .apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied threshold-outside ([{},{}], outside={}) to {} -> {}",
        args.band.lower_threshold,
        args.band.upper_threshold,
        args.band.outside_value,
        args.input.display(),
        args.output.display()
    );
    info!("filter: threshold-outside complete");

    Ok(())
}

pub(super) fn run_sigmoid(args: &FilterArgs) -> Result<()> {
    use ritk_filter::SigmoidImageFilter;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_native_read_capable(input_format),
        "sigmoid requires native input support for {:?}",
        input_format
    );
    anyhow::ensure!(
        is_native_write_capable(output_format),
        "sigmoid requires native output support for {:?}",
        output_format
    );
    let image = read_image_native(&args.input)?;
    let backend = NativeBackend::default();
    let filtered = SigmoidImageFilter::new(
        args.sigmoid.midpoint,
        args.sigmoid.steepness,
        args.range.out_min,
        args.range.out_max,
    )
    .apply_native(&image, &backend)?;

    write_image_native(&args.output, &filtered, output_format)?;

    println!(
        "Applied sigmoid (midpoint={}, steepness={}, out=[{},{}]) to {} -> {}",
        args.sigmoid.midpoint,
        args.sigmoid.steepness,
        args.range.out_min,
        args.range.out_max,
        args.input.display(),
        args.output.display()
    );
    info!("filter: sigmoid complete");

    Ok(())
}

pub(super) fn run_binary_threshold(args: &FilterArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let filtered = ritk_segmentation::binary_threshold(
        &image,
        args.band.lower_threshold,
        args.band.upper_threshold,
        args.band.foreground_value,
        args.band.background_value,
    );

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied binary-threshold ([{},{}] fg={} bg={}) to {} -> {}",
        args.band.lower_threshold,
        args.band.upper_threshold,
        args.band.foreground_value,
        args.band.background_value,
        args.input.display(),
        args.output.display()
    );
    info!("filter: binary-threshold complete");

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::filter::{default_args, make_test_image, FilterKind};
    use ritk_core::image::Image;
    use ritk_image::tensor::Backend as BurnBackend;
    use ritk_image::tensor::{Shape, Tensor, TensorData};
    use ritk_spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    #[test]
    fn test_filter_bed_separation_masks_background() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("filtered.mha");

        let values = vec![
            -1000.0, -1000.0, -1000.0, -1000.0, -200.0, -150.0, 50.0, 60.0, -1000.0, -1000.0,
            -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0,
        ];
        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(values, Shape::new([1, 4, 4]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        ritk_io::write_metaimage(&input, &image).unwrap();

        let mut args = default_args(input.clone(), output.clone(), FilterKind::BedSeparation);
        args.bed.body_threshold = -350.0;
        args.bed.closing_radius = 0;
        args.bed.opening_radius = 0;
        args.bed.outside_value = -2048.0;
        super::run_bed_separation(&args).unwrap();

        let result = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
        let result_data = result.data_slice();
        assert_eq!(result_data[0], -2048.0);
        assert_eq!(result_data[1], -2048.0);
        assert_eq!(result_data[4], -200.0);
        assert_eq!(result_data[5], -150.0);
        assert_eq!(result_data[6], 50.0);
        assert_eq!(result_data[7], 60.0);
    }

    #[test]
    fn test_filter_rescale_intensity_output_range() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let args = default_args(input, output.clone(), FilterKind::RescaleIntensity);
        run_rescale_intensity(&args).unwrap();
        assert!(output.exists());

        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        result.with_data_slice(|vals| {
            let min_val = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                (min_val - 0.0).abs() < 1e-4,
                "rescale-intensity min must be 0.0, got {}",
                min_val
            );
            assert!(
                (max_val - 1.0).abs() < 1e-4,
                "rescale-intensity max must be 1.0, got {}",
                max_val
            );
        });
    }

    #[test]
    fn test_filter_intensity_windowing_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), FilterKind::IntensityWindowing);
        args.window.window_min = 20.0;
        args.window.window_max = 80.0;
        args.range.out_min = 0.0;
        args.range.out_max = 1.0;
        run_intensity_windowing(&args).unwrap();
        assert!(output.exists());

        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(result.shape(), [5, 5, 5]);
    }

    #[test]
    fn test_filter_threshold_below_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), FilterKind::ThresholdBelow);
        args.threshold.threshold_value = 50.0;
        args.band.outside_value = 0.0;
        run_threshold_below(&args).unwrap();
        assert!(output.exists());

        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        result.with_data_slice(|vals| {
            // All pixels that were < 50 should now be 0.0
            // Original values are 0..124, so values 0..49 -> 0.0
            let count_zero = vals.iter().filter(|&&v| v == 0.0).count();
            assert!(
                count_zero >= 50,
                "at least 50 pixels should be zeroed, got {}",
                count_zero
            );
        });
    }

    #[test]
    fn test_filter_threshold_above_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), FilterKind::ThresholdAbove);
        args.threshold.threshold_value = 50.0;
        args.band.outside_value = 0.0;
        run_threshold_above(&args).unwrap();
        assert!(output.exists());

        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(result.shape(), [5, 5, 5]);
    }

    #[test]
    fn test_filter_threshold_outside_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), FilterKind::ThresholdOutside);
        args.band.lower_threshold = 30.0;
        args.band.upper_threshold = 90.0;
        args.band.outside_value = 0.0;
        run_threshold_outside(&args).unwrap();
        assert!(output.exists());

        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(result.shape(), [5, 5, 5]);
    }

    #[test]
    fn test_filter_sigmoid_creates_output_bounded() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), FilterKind::Sigmoid);
        args.sigmoid.midpoint = 62.0; // midpoint of 0..124
        args.sigmoid.steepness = 20.0;
        args.range.out_min = 0.0;
        args.range.out_max = 1.0;
        run_sigmoid(&args).unwrap();
        assert!(output.exists());

        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        result.with_data_slice(|vals| {
            for &v in vals {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "sigmoid output must be in [0,1], got {}",
                    v
                );
            }
        });
    }

    #[test]
    fn test_filter_binary_threshold_produces_binary_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), FilterKind::BinaryThreshold);
        args.band.lower_threshold = 40.0;
        args.band.upper_threshold = 80.0;
        args.band.foreground_value = 1.0;
        args.band.background_value = 0.0;
        run_binary_threshold(&args).unwrap();
        assert!(output.exists());

        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        result.with_data_slice(|vals| {
            for &v in vals {
                assert!(
                    v == 0.0 || v == 1.0,
                    "binary-threshold output must be 0.0 or 1.0, got {}",
                    v
                );
            }
        });
    }
}
