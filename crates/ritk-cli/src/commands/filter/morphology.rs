use anyhow::Result;
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

pub(super) fn run_grayscale_erosion(args: &FilterArgs) -> Result<()> {
    use ritk_filter::GrayscaleErosion;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow::anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_native_read_capable(input_format) && is_native_write_capable(output_format),
        "grayscale-erosion requires native input/output formats"
    );
    let image = read_image_native(&args.input)?;
    let backend = NativeBackend::default();
    let filtered = GrayscaleErosion::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image_native(&args.output, &filtered, output_format)?;
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
        is_native_read_capable(input_format) && is_native_write_capable(output_format),
        "grayscale-dilation requires native input/output formats"
    );
    let image = read_image_native(&args.input)?;
    let backend = NativeBackend::default();
    let filtered = GrayscaleDilation::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image_native(&args.output, &filtered, output_format)?;
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
        is_native_read_capable(input_format) && is_native_write_capable(output_format),
        "white-top-hat requires native input/output formats"
    );
    let image = read_image_native(&args.input)?;
    let backend = NativeBackend::default();
    let filtered = WhiteTopHatFilter::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image_native(&args.output, &filtered, output_format)?;
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
        is_native_read_capable(input_format) && is_native_write_capable(output_format),
        "black-top-hat requires native input/output formats"
    );
    let image = read_image_native(&args.input)?;
    let backend = NativeBackend::default();
    let filtered = BlackTopHatFilter::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image_native(&args.output, &filtered, output_format)?;
    info!("filter: black-top-hat complete");

    Ok(())
}

pub(super) fn run_hit_or_miss(args: &FilterArgs) -> Result<()> {
    use ritk_filter::HitOrMissTransform;

    let image = read_image(&args.input)?;
    let filtered = HitOrMissTransform::new(args.kernel.radius, args.kernel.radius).apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;
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
        is_native_read_capable(input_format) && is_native_write_capable(output_format),
        "label-dilation requires native input/output formats"
    );
    let image = read_image_native(&args.input)?;
    let backend = NativeBackend::default();
    let filtered = LabelDilation::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image_native(&args.output, &filtered, output_format)?;
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
        is_native_read_capable(input_format) && is_native_write_capable(output_format),
        "label-erosion requires native input/output formats"
    );
    let image = read_image_native(&args.input)?;
    let backend = NativeBackend::default();
    let filtered = LabelErosion::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image_native(&args.output, &filtered, output_format)?;
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
        is_native_read_capable(input_format) && is_native_write_capable(output_format),
        "label-opening requires native input/output formats"
    );
    let image = read_image_native(&args.input)?;
    let backend = NativeBackend::default();
    let filtered = LabelOpening::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image_native(&args.output, &filtered, output_format)?;
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
        is_native_read_capable(input_format) && is_native_write_capable(output_format),
        "label-closing requires native input/output formats"
    );
    let image = read_image_native(&args.input)?;
    let backend = NativeBackend::default();
    let filtered = LabelClosing::new(args.kernel.radius).apply_native(&image, &backend)?;

    write_image_native(&args.output, &filtered, output_format)?;
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
        is_native_read_capable(input_format)
            && is_native_read_capable(mask_format)
            && is_native_write_capable(output_format),
        "morphological-reconstruction requires native marker, mask, and output formats"
    );
    let marker = read_image_native(&args.input)?;
    let mask = read_image_native(mask_path)?;
    let backend = NativeBackend::default();

    let filtered = MorphologicalReconstruction::new(ReconstructionMode::Dilation)
        .apply_native(&marker, &mask, &backend)?;

    write_image_native(&args.output, &filtered, output_format)?;
    info!("filter: morphological-reconstruction complete");

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
    fn grayscale_erosion_writes_native_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("output.nii");
        ritk_io::write_nifti(&input, &make_test_image()).expect("input fixture writes");

        let mut args = default_args(input, output.clone(), FilterKind::GrayscaleErosion);
        args.kernel.radius = 1;
        run_grayscale_erosion(&args).expect("grayscale erosion succeeds");
        let output = crate::commands::read_image_native(&output)
            .expect("grayscale erosion output is natively readable");
        assert_eq!(output.shape(), [5, 5, 5]);
    }

    #[test]
    fn grayscale_dilation_writes_native_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("output.nii");
        ritk_io::write_nifti(&input, &make_test_image()).expect("input fixture writes");

        let mut args = default_args(input, output.clone(), FilterKind::GrayscaleDilation);
        args.kernel.radius = 1;
        run_grayscale_dilation(&args).expect("grayscale dilation succeeds");
        let output = crate::commands::read_image_native(&output)
            .expect("grayscale dilation output is natively readable");
        assert_eq!(output.shape(), [5, 5, 5]);
    }

    #[test]
    fn top_hat_routes_write_native_output() {
        let dir = tempdir().unwrap();
        for kind in [FilterKind::WhiteTopHat, FilterKind::BlackTopHat] {
            let input = dir.path().join(format!("{kind:?}-input.nii"));
            let output = dir.path().join(format!("{kind:?}-output.nii"));
            ritk_io::write_nifti(&input, &make_test_image()).expect("input fixture writes");

            let mut args = default_args(input, output.clone(), kind);
            args.kernel.radius = 1;
            match kind {
                FilterKind::WhiteTopHat => run_white_top_hat(&args),
                FilterKind::BlackTopHat => run_black_top_hat(&args),
                _ => unreachable!("top-hat test only enumerates top-hat routes"),
            }
            .expect("top-hat route succeeds");
            let output = crate::commands::read_image_native(&output)
                .expect("top-hat output is natively readable");
            assert_eq!(output.shape(), [5, 5, 5]);
        }
    }

    #[test]
    fn morphological_reconstruction_writes_native_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("marker.nii");
        let mask = dir.path().join("mask.nii");
        let output = dir.path().join("output.nii");
        let fixture = make_test_image();
        ritk_io::write_nifti(&input, &fixture).expect("marker fixture writes");
        ritk_io::write_nifti(&mask, &fixture).expect("mask fixture writes");

        let mut args = default_args(
            input,
            output.clone(),
            FilterKind::MorphologicalReconstruction,
        );
        args.mask_input.mask = Some(mask);
        run_morphological_reconstruction(&args).expect("morphological reconstruction succeeds");
        let output = crate::commands::read_image_native(&output)
            .expect("reconstruction output is natively readable");
        assert_eq!(output.shape(), [5, 5, 5]);
    }

    #[test]
    fn label_dilation_writes_native_values() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");
        let device: <Backend as BurnBackend>::Device = Default::default();
        let mut values = vec![0.0f32; 125];
        values[2 * 25 + 2 * 5 + 2] = 3.0;
        let tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(values, Shape::new([5, 5, 5])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        ritk_io::write_nifti::<Backend, _>(&input_path, &image).expect("input fixture writes");

        let mut args = default_args(input_path, output_path.clone(), FilterKind::LabelDilation);
        args.kernel.radius = 1;
        run_label_dilation(&args).expect("label dilation succeeds");
        let output = crate::commands::read_image_native(&output_path)
            .expect("label dilation output is natively readable");
        let values = output.data_slice().expect("contiguous native output");
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

        let device: <Backend as BurnBackend>::Device = Default::default();
        let mut v = vec![0.0f32; 125];
        v[2 * 25 + 2 * 5 + 2] = 1.0;
        let td = TensorData::new(v, Shape::new([5, 5, 5]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        ritk_io::write_nifti::<Backend, _>(&input_path, &image).unwrap();

        let mut args = default_args(input_path, output_path.clone(), FilterKind::LabelErosion);
        args.kernel.radius = 1;
        run_label_erosion(&args).expect("label-erosion must succeed");
        let output = crate::commands::read_image_native(&output_path)
            .expect("label erosion output is natively readable");
        assert_eq!(output.shape(), [5, 5, 5]);
        assert_eq!(
            output.data_slice().expect("contiguous native output")[2 * 25 + 2 * 5 + 2],
            0.0
        );
    }

    #[test]
    fn test_filter_label_opening_creates_output() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");

        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![1.0f32; 125], Shape::new([5, 5, 5]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        ritk_io::write_nifti::<Backend, _>(&input_path, &image).unwrap();

        let mut args = default_args(input_path, output_path.clone(), FilterKind::LabelOpening);
        args.kernel.radius = 1;
        run_label_opening(&args).expect("label-opening must succeed");
        let output = crate::commands::read_image_native(&output_path)
            .expect("label opening output is natively readable");
        assert_eq!(output.shape(), [5, 5, 5]);
    }

    #[test]
    fn test_filter_label_closing_creates_output() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");

        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![1.0f32; 125], Shape::new([5, 5, 5]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        ritk_io::write_nifti::<Backend, _>(&input_path, &image).unwrap();

        let mut args = default_args(input_path, output_path.clone(), FilterKind::LabelClosing);
        args.kernel.radius = 1;
        run_label_closing(&args).expect("label-closing must succeed");
        let output = crate::commands::read_image_native(&output_path)
            .expect("label closing output is natively readable");
        assert_eq!(output.shape(), [5, 5, 5]);
    }

    #[test]
    fn test_filter_morph_recon_requires_mask() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");

        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![0.5f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        ritk_io::write_nifti::<Backend, _>(&input_path, &image).unwrap();

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
