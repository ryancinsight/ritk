use anyhow::Result;
use tracing::info;

use super::{
    super::{infer_format, is_read_capable, is_write_capable, read_image, write_image, Backend},
    FilterArgs,
};

pub(super) fn run_grayscale_erosion(args: &FilterArgs) -> Result<()> {
    use ritk_filter::GrayscaleErosion;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_read_capable(input_format) && is_write_capable(output_format),
        "grayscale-erosion requires native input/output formats"
    );
    let image = read_image(&args.input)?;
    let backend = Backend::default();
    let filtered = GrayscaleErosion::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image(&args.output, &filtered, output_format)?;
    info!("filter: grayscale-erosion complete");

    Ok(())
}

pub(super) fn run_grayscale_dilation(args: &FilterArgs) -> Result<()> {
    use ritk_filter::GrayscaleDilation;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_read_capable(input_format) && is_write_capable(output_format),
        "grayscale-dilation requires native input/output formats"
    );
    let image = read_image(&args.input)?;
    let backend = Backend::default();
    let filtered = GrayscaleDilation::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image(&args.output, &filtered, output_format)?;
    info!("filter: grayscale-dilation complete");

    Ok(())
}

pub(super) fn run_white_top_hat(args: &FilterArgs) -> Result<()> {
    use ritk_filter::WhiteTopHatFilter;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_read_capable(input_format) && is_write_capable(output_format),
        "white-top-hat requires native input/output formats"
    );
    let image = read_image(&args.input)?;
    let backend = Backend::default();
    let filtered = WhiteTopHatFilter::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image(&args.output, &filtered, output_format)?;
    info!("filter: white-top-hat complete");

    Ok(())
}

pub(super) fn run_black_top_hat(args: &FilterArgs) -> Result<()> {
    use ritk_filter::BlackTopHatFilter;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_read_capable(input_format) && is_write_capable(output_format),
        "black-top-hat requires native input/output formats"
    );
    let image = read_image(&args.input)?;
    let backend = Backend::default();
    let filtered = BlackTopHatFilter::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image(&args.output, &filtered, output_format)?;
    info!("filter: black-top-hat complete");

    Ok(())
}

pub(super) fn run_hit_or_miss(args: &FilterArgs) -> Result<()> {
    use ritk_filter::HitOrMissTransform;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_read_capable(input_format) && is_write_capable(output_format),
        "hit-or-miss requires native input/output formats"
    );
    let image = read_image(&args.input)?;
    let backend = Backend::default();
    let filtered = HitOrMissTransform::new(args.kernel.radius, args.kernel.radius)
        .apply_native(&image, &backend)?;

    write_image(&args.output, &filtered, output_format)?;
    info!("filter: hit-or-miss complete");

    Ok(())
}

pub(super) fn run_label_dilation(args: &FilterArgs) -> Result<()> {
    use ritk_filter::LabelDilation;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_read_capable(input_format) && is_write_capable(output_format),
        "label-dilation requires native input/output formats"
    );
    let image = read_image(&args.input)?;
    let backend = Backend::default();
    let filtered = LabelDilation::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image(&args.output, &filtered, output_format)?;
    info!(
        "filter: label-dilation complete radius={} input={} output={}",
        args.kernel.radius,
        args.input.display(),
        args.output.display()
    );

    Ok(())
}

pub(super) fn run_label_erosion(args: &FilterArgs) -> Result<()> {
    use ritk_filter::LabelErosion;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_read_capable(input_format) && is_write_capable(output_format),
        "label-erosion requires native input/output formats"
    );
    let image = read_image(&args.input)?;
    let backend = Backend::default();
    let filtered = LabelErosion::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image(&args.output, &filtered, output_format)?;
    info!("filter: label-erosion complete");

    Ok(())
}

pub(super) fn run_label_opening(args: &FilterArgs) -> Result<()> {
    use ritk_filter::LabelOpening;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_read_capable(input_format) && is_write_capable(output_format),
        "label-opening requires native input/output formats"
    );
    let image = read_image(&args.input)?;
    let backend = Backend::default();
    let filtered = LabelOpening::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image(&args.output, &filtered, output_format)?;
    info!("filter: label-opening complete");

    Ok(())
}

pub(super) fn run_label_closing(args: &FilterArgs) -> Result<()> {
    use ritk_filter::LabelClosing;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_read_capable(input_format) && is_write_capable(output_format),
        "label-closing requires native input/output formats"
    );
    let image = read_image(&args.input)?;
    let backend = Backend::default();
    let filtered = LabelClosing::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image(&args.output, &filtered, output_format)?;
    info!("filter: label-closing complete");

    Ok(())
}

pub(super) fn run_morphological_reconstruction(args: &FilterArgs) -> Result<()> {
    use ritk_filter::{MorphologicalReconstruction, ReconstructionMode};

    let mask_path =
        args.mask_input.mask.as_ref().ok_or_else(|| {
            anyhow::anyhow!("morphological-reconstruction requires --mask <path>")
        })?;
    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let mask_format = infer_format(mask_path)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer mask format: {}", mask_path.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_read_capable(input_format)
            && is_read_capable(mask_format)
            && is_write_capable(output_format),
        "morphological-reconstruction requires native marker, mask, and output formats"
    );
    let marker = read_image(&args.input)?;
    let mask = read_image(mask_path)?;
    let backend = Backend::default();

    let filtered = MorphologicalReconstruction::new(ReconstructionMode::Dilation)
        .apply_native(&marker, &mask, &backend)?;

    write_image(&args.output, &filtered, output_format)?;
    info!("filter: morphological-reconstruction complete");

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::filter::{default_args, make_test_image, FilterKind};
    use crate::commands::{read_image, write_image, Backend};
    use ritk_image::Image;
    use ritk_io::ImageFormat;
    use ritk_spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    #[test]
    fn grayscale_erosion_writes_native_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("output.nii");
        write_image(&input, &make_test_image(), ImageFormat::NIfTI).expect("input fixture writes");

        let mut args = default_args(input, output.clone(), FilterKind::GrayscaleErosion);
        args.kernel.radius = 1;
        run_grayscale_erosion(&args).expect("grayscale erosion succeeds");
        let output = read_image(&output).expect("grayscale erosion output is natively readable");
        assert_eq!(output.shape(), [5, 5, 5]);
    }

    #[test]
    fn grayscale_dilation_writes_native_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("output.nii");
        write_image(&input, &make_test_image(), ImageFormat::NIfTI).expect("input fixture writes");

        let mut args = default_args(input, output.clone(), FilterKind::GrayscaleDilation);
        args.kernel.radius = 1;
        run_grayscale_dilation(&args).expect("grayscale dilation succeeds");
        let output = read_image(&output).expect("grayscale dilation output is natively readable");
        assert_eq!(output.shape(), [5, 5, 5]);
    }

    #[test]
    fn top_hat_routes_write_native_output() {
        let dir = tempdir().unwrap();
        for kind in [FilterKind::WhiteTopHat, FilterKind::BlackTopHat] {
            let input = dir.path().join(format!("{kind:?}-input.nii"));
            let output = dir.path().join(format!("{kind:?}-output.nii"));
            write_image(&input, &make_test_image(), ImageFormat::NIfTI)
                .expect("input fixture writes");

            let mut args = default_args(input, output.clone(), kind);
            args.kernel.radius = 1;
            match kind {
                FilterKind::WhiteTopHat => run_white_top_hat(&args),
                FilterKind::BlackTopHat => run_black_top_hat(&args),
                _ => unreachable!("top-hat test only enumerates top-hat routes"),
            }
            .expect("top-hat route succeeds");
            let output = read_image(&output).expect("top-hat output is natively readable");
            assert_eq!(output.shape(), [5, 5, 5]);
        }
    }

    #[test]
    fn hit_or_miss_writes_exact_native_values_and_geometry() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");
        let values: Vec<f32> = (0..27)
            .map(|index| if index % 4 == 0 { 1.0 } else { 0.0 })
            .collect();
        let origin = Point::new([2.0, 3.0, 5.0]);
        let spacing = Spacing::new([0.5, 1.0, 2.0]);
        let native = Image::<f32, Backend, 3>::from_flat_on(
            values.clone(),
            [3, 3, 3],
            origin,
            spacing,
            Direction::identity(),
            &Backend::default(),
        )
        .expect("invariant: valid hit-or-miss fixture");
        write_image(&input_path, &native, ImageFormat::NIfTI).expect("input fixture writes");

        let mut args = default_args(input_path, output_path.clone(), FilterKind::HitOrMiss);
        args.kernel.radius = 0;
        run_hit_or_miss(&args).expect("hit-or-miss succeeds");
        let output = read_image(&output_path).expect("hit-or-miss output is natively readable");

        assert_eq!(output.shape(), [3, 3, 3]);
        assert_eq!(*output.origin(), origin);
        assert_eq!(*output.spacing(), spacing);
        assert_eq!(
            output
                .data_slice()
                .expect("invariant: image storage is contiguous"),
            values
        );
    }

    #[test]
    fn morphological_reconstruction_writes_native_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("marker.nii");
        let mask = dir.path().join("mask.nii");
        let output = dir.path().join("output.nii");
        let fixture = make_test_image();
        write_image(&input, &fixture, ImageFormat::NIfTI).expect("marker fixture writes");
        write_image(&mask, &fixture, ImageFormat::NIfTI).expect("mask fixture writes");

        let mut args = default_args(
            input,
            output.clone(),
            FilterKind::MorphologicalReconstruction,
        );
        args.mask_input.mask = Some(mask);
        run_morphological_reconstruction(&args).expect("morphological reconstruction succeeds");
        let output = read_image(&output).expect("reconstruction output is natively readable");
        assert_eq!(output.shape(), [5, 5, 5]);
    }

    #[test]
    fn label_dilation_writes_native_values() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");
        let mut values = vec![0.0f32; 125];
        values[2 * 25 + 2 * 5 + 2] = 3.0;
        let native = Image::<f32, Backend, 3>::from_flat_on(
            values,
            [5, 5, 5],
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
            &Backend::default(),
        )
        .expect("invariant: valid label dilation fixture");
        write_image(&input_path, &native, ImageFormat::NIfTI).expect("input fixture writes");

        let mut args = default_args(input_path, output_path.clone(), FilterKind::LabelDilation);
        args.kernel.radius = 1;
        run_label_dilation(&args).expect("label dilation succeeds");
        let output = read_image(&output_path).expect("label dilation output is natively readable");
        let values = output
            .data_slice()
            .expect("invariant: image storage is contiguous");
        assert_eq!(output.shape(), [5, 5, 5]);
        assert_eq!(values[2 * 25 + 2 * 5 + 2], 3.0);
        assert_eq!(values[2 * 25 + 2 * 5 + 1], 3.0);
        assert_eq!(values[0], 0.0);
    }

    #[test]
    fn test_filter_label_erosion_creates_output() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");

        let mut v = vec![0.0f32; 125];
        v[2 * 25 + 2 * 5 + 2] = 1.0;
        let native = Image::<f32, Backend, 3>::from_flat_on(
            v,
            [5, 5, 5],
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
            &Backend::default(),
        )
        .expect("invariant: valid label erosion fixture");
        write_image(&input_path, &native, ImageFormat::NIfTI).unwrap();

        let mut args = default_args(input_path, output_path.clone(), FilterKind::LabelErosion);
        args.kernel.radius = 1;
        run_label_erosion(&args).expect("label-erosion must succeed");
        let output = read_image(&output_path).expect("label erosion output is natively readable");
        assert_eq!(output.shape(), [5, 5, 5]);
        assert_eq!(
            output
                .data_slice()
                .expect("invariant: image storage is contiguous")[2 * 25 + 2 * 5 + 2],
            0.0
        );
    }

    #[test]
    fn test_filter_label_opening_creates_output() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");

        let native = Image::<f32, Backend, 3>::from_flat_on(
            vec![1.0f32; 125],
            [5, 5, 5],
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
            &Backend::default(),
        )
        .expect("invariant: valid label opening fixture");
        write_image(&input_path, &native, ImageFormat::NIfTI).unwrap();

        let mut args = default_args(input_path, output_path.clone(), FilterKind::LabelOpening);
        args.kernel.radius = 1;
        run_label_opening(&args).expect("label-opening must succeed");
        let output = read_image(&output_path).expect("label opening output is natively readable");
        assert_eq!(output.shape(), [5, 5, 5]);
    }

    #[test]
    fn test_filter_label_closing_creates_output() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");

        let native = Image::<f32, Backend, 3>::from_flat_on(
            vec![1.0f32; 125],
            [5, 5, 5],
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
            &Backend::default(),
        )
        .expect("invariant: valid label closing fixture");
        write_image(&input_path, &native, ImageFormat::NIfTI).unwrap();

        let mut args = default_args(input_path, output_path.clone(), FilterKind::LabelClosing);
        args.kernel.radius = 1;
        run_label_closing(&args).expect("label-closing must succeed");
        let output = read_image(&output_path).expect("label closing output is natively readable");
        assert_eq!(output.shape(), [5, 5, 5]);
    }

    #[test]
    fn test_filter_morph_recon_requires_mask() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");

        let image = Image::<f32, Backend, 3>::from_flat_on(
            vec![0.5; 8],
            [2, 2, 2],
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
            &Backend::default(),
        )
        .expect("invariant: fixture tensor has the declared rank");
        write_image(&input_path, &image, ImageFormat::NIfTI).unwrap();

        let mut args = default_args(
            input_path,
            output_path,
            FilterKind::MorphologicalReconstruction,
        );
        args.mask_input.mask = None;
        let result = run_morphological_reconstruction(&args);
        assert!(result.is_err(), "missing mask must return Err");
        assert!(result.unwrap_err().to_string().contains("mask"));
    }
}
