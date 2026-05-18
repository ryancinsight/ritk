use anyhow::Result;
use tracing::info;

#[cfg(test)]
use super::Backend;
use super::{read_image, write_image_inferred, FilterArgs};

pub(super) fn run_grayscale_erosion(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::GrayscaleErosion;

    let image = read_image(&args.input)?;
    let filtered = GrayscaleErosion::new(args.radius).apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;
    info!("filter: grayscale-erosion complete");

    Ok(())
}

pub(super) fn run_grayscale_dilation(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::GrayscaleDilation;

    let image = read_image(&args.input)?;
    let filtered = GrayscaleDilation::new(args.radius).apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;
    info!("filter: grayscale-dilation complete");

    Ok(())
}

pub(super) fn run_white_top_hat(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::WhiteTopHatFilter;

    let image = read_image(&args.input)?;
    let filtered = WhiteTopHatFilter::new(args.radius).apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;
    info!("filter: white-top-hat complete");

    Ok(())
}

pub(super) fn run_black_top_hat(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::BlackTopHatFilter;

    let image = read_image(&args.input)?;
    let filtered = BlackTopHatFilter::new(args.radius).apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;
    info!("filter: black-top-hat complete");

    Ok(())
}

pub(super) fn run_hit_or_miss(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::HitOrMissTransform;

    let image = read_image(&args.input)?;
    let filtered = HitOrMissTransform::new(args.radius, args.radius).apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;
    info!("filter: label-dilation complete");

    Ok(())
}

pub(super) fn run_label_dilation(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::LabelDilation;

    let image = read_image(&args.input)?;
    let filtered = LabelDilation::new(args.radius).apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;
    info!(
        "filter: label-dilation complete radius={} input={} output={}",
        args.radius,
        args.input.display(),
        args.output.display()
    );

    Ok(())
}

pub(super) fn run_label_erosion(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::LabelErosion;

    let image = read_image(&args.input)?;
    let filtered = LabelErosion::new(args.radius).apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;
    info!("filter: label-erosion complete");

    Ok(())
}

pub(super) fn run_label_opening(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::LabelOpening;

    let image = read_image(&args.input)?;
    let filtered = LabelOpening::new(args.radius).apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;
    info!("filter: label-opening complete");

    Ok(())
}

pub(super) fn run_label_closing(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::LabelClosing;

    let image = read_image(&args.input)?;
    let filtered = LabelClosing::new(args.radius).apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;
    info!("filter: label-closing complete");

    Ok(())
}

pub(super) fn run_morphological_reconstruction(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::{MorphologicalReconstruction, ReconstructionMode};

    let marker = read_image(&args.input)?;
    let mask_path = args
        .mask
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("morphological-reconstruction requires --mask <path>"))?;
    let mask = read_image(mask_path)?;

    let filtered =
        MorphologicalReconstruction::new(ReconstructionMode::Dilation).apply(&marker, &mask)?;

    write_image_inferred(&args.output, &filtered)?;
    info!("filter: morphological-reconstruction complete");

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::filter::default_args;
    use burn::tensor::backend::Backend as BurnBackend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

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

        let mut args = default_args(input_path, output_path.clone(), "label-erosion");
        args.radius = 1;
        run_label_erosion(&args).expect("label-erosion must succeed");
        assert!(output_path.exists());
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

        let mut args = default_args(input_path, output_path.clone(), "label-opening");
        args.radius = 1;
        run_label_opening(&args).expect("label-opening must succeed");
        assert!(output_path.exists());
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

        let mut args = default_args(input_path, output_path.clone(), "label-closing");
        args.radius = 1;
        run_label_closing(&args).expect("label-closing must succeed");
        assert!(output_path.exists());
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

        let mut args = default_args(input_path, output_path, "morphological-reconstruction");
        args.mask = None;
        let result = run_morphological_reconstruction(&args);
        assert!(result.is_err(), "missing mask must return Err");
        assert!(result.unwrap_err().to_string().contains("mask"));
    }
}
