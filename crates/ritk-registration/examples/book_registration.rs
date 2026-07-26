//! Generate the real CT/MR registration figure used by the RITK mdBook.
//!
//! The example loads the in-tree RIRE Patient 001 CT and MR T1 volumes,
//! evaluates the classical mutual-information metric on a coarse common grid,
//! and resamples the original MR volume onto the full CT grid with the
//! dataset's fiducial CT-to-MR transform. The same transform is shown beside
//! the RIRE reference panel so the rendered result is checked against the
//! supplied registration standard.

use anyhow::{bail, Context, Result};
use coeus_core::SequentialBackend;
use eunomia::CastFrom;
use image::{Rgb, RgbImage};
use ritk_filter::resample::native::{fixed_world_points, resample_moving_at_world};
use ritk_image::Image;
use ritk_io::{format::metaimage::native::MetaImageReader, ImageReader};
use ritk_registration::classical::{engine::MutualInformationMetric, image_to_leto_volume};
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::transform::affine::AtlasAffineTransform;
use std::path::{Path, PathBuf};

type Backend = SequentialBackend;

const CT_PATH: &str = "test_data/registration/rire/training_001_ct.mha";
const MR_PATH: &str = "test_data/registration/rire/training_001_mr_T1.mha";
const GROUND_TRUTH_PATH: &str =
    "test_data/registration/rire/training_001_ct_to_mr_T1_ground_truth.tfm";
const COARSE_ORIGIN: [f64; 3] = [35.0, 40.0, 40.0];
const COARSE_EXTENT: [f64; 3] = [65.0, 240.0, 240.0];
const COARSE_SHAPE: [usize; 3] = [8, 64, 64];
const DISPLAY_WIDTH: u32 = 256;
const DISPLAY_HEIGHT: u32 = 256;
const PANEL_GAP: u32 = 8;

fn read_inputs() -> Result<(Image<f32, Backend, 3>, Image<f32, Backend, 3>)> {
    let reader = MetaImageReader::new(Backend::default());
    let ct = reader
        .read(CT_PATH)
        .with_context(|| format!("read RIRE CT volume {CT_PATH}"))?;
    let mr = reader
        .read(MR_PATH)
        .with_context(|| format!("read RIRE MR volume {MR_PATH}"))?;
    Ok((ct, mr))
}

fn coarse_grid() -> Result<Image<f32, Backend, 3>> {
    let spacing = std::array::from_fn(|axis| {
        COARSE_EXTENT[axis] / (COARSE_SHAPE[axis].saturating_sub(1) as f64)
    });
    let voxel_count = COARSE_SHAPE
        .into_iter()
        .try_fold(1_usize, usize::checked_mul)
        .context("coarse registration grid size overflows usize")?;
    Image::from_flat(
        vec![0.0; voxel_count],
        COARSE_SHAPE,
        Point::new(COARSE_ORIGIN),
        Spacing::new(spacing),
        Direction::identity(),
    )
    .map_err(|error| anyhow::anyhow!("construct coarse registration grid: {error}"))
}

fn image_from_grid(
    values: Vec<f32>,
    grid: &Image<f32, Backend, 3>,
) -> Result<Image<f32, Backend, 3>> {
    Image::from_flat(
        values,
        grid.shape(),
        *grid.origin(),
        *grid.spacing(),
        *grid.direction(),
    )
    .map_err(|error| anyhow::anyhow!("construct resampled registration image: {error}"))
}

fn percentile(values: &[f32], hundredths: usize) -> Result<f32> {
    let mut finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        bail!("cannot compute an intensity percentile from an empty image");
    }
    finite.sort_by(f32::total_cmp);
    let index = finite.len().saturating_mul(hundredths) / 100;
    finite
        .get(index.min(finite.len() - 1))
        .copied()
        .context("percentile index is outside the sorted intensity range")
}

fn window(values: &[f32], lower: f32, upper: f32) -> Result<Vec<f32>> {
    if !lower.is_finite()
        || !upper.is_finite()
        || !matches!(lower.partial_cmp(&upper), Some(std::cmp::Ordering::Less))
    {
        bail!("intensity window must be finite and strictly increasing");
    }
    Ok(values
        .iter()
        .map(|value| ((*value - lower) / (upper - lower)).clamp(0.0, 1.0) * 255.0)
        .collect())
}

fn ground_truth_transform() -> Result<AtlasAffineTransform<Backend, 3>> {
    let source = std::fs::read_to_string(GROUND_TRUTH_PATH)
        .with_context(|| format!("read RIRE ground-truth transform {GROUND_TRUTH_PATH}"))?;
    let parameters = source
        .lines()
        .find_map(|line| line.strip_prefix("Parameters:"))
        .context("RIRE transform does not contain a Parameters line")?
        .split_whitespace()
        .map(str::parse::<f32>)
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("parse RIRE Euler3D parameters")?;
    let [angle_x, angle_y, angle_z, tx, ty, tz] = parameters.as_slice() else {
        bail!(
            "RIRE Euler3D transform must contain six parameters, got {}",
            parameters.len()
        );
    };
    let (cx, sx) = (angle_x.cos(), angle_x.sin());
    let (cy, sy) = (angle_y.cos(), angle_y.sin());
    let (cz, sz) = (angle_z.cos(), angle_z.sin());
    let rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
    let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
    let matrix: [[f32; 3]; 3] = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|inner| rz[row][inner] * rx[inner][column])
                .sum::<f32>()
        })
    });
    let matrix: [[f32; 3]; 3] = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|inner| matrix[row][inner] * ry[inner][column])
                .sum::<f32>()
        })
    });
    let matrix = matrix.into_iter().flatten().collect::<Vec<_>>();
    let translation = [*tx, *ty, *tz];
    AtlasAffineTransform::try_new(&matrix, &translation, &[0.0; 3])
        .map_err(|error| anyhow::anyhow!("construct RIRE ground-truth transform: {error}"))
}

fn slice(values: &[f32], shape: [usize; 3], z: usize) -> Result<&[f32]> {
    let [depth, height, width] = shape;
    if z >= depth {
        bail!("axial slice {z} is outside volume depth {depth}");
    }
    let plane = height
        .checked_mul(width)
        .context("axial plane size overflows usize")?;
    let start = z
        .checked_mul(plane)
        .context("axial slice offset overflows usize")?;
    values
        .get(start..start + plane)
        .context("image data length does not match its declared shape")
}

fn normalized_slice(values: &[f32], shape: [usize; 3], z: usize) -> Result<Vec<f32>> {
    let values = slice(values, shape, z)?;
    let lower = percentile(values, 2)?;
    let upper = percentile(values, 98)?;
    window(values, lower, upper)
}

fn normalized_ct_slice(values: &[f32], shape: [usize; 3], z: usize) -> Result<Vec<f32>> {
    window(slice(values, shape, z)?, -1000.0, 1000.0)
}

fn normalized_to_u8(value: f32) -> u8 {
    u8::cast_from((value.clamp(0.0, 255.0) / 255.0) * 255.0)
}

fn render_panel(
    figure: &mut RgbImage,
    panel_index: usize,
    fixed: &[f32],
    moving: Option<&[f32]>,
    shape: [usize; 3],
    axial_slice: usize,
) -> Result<()> {
    let fixed = slice(fixed, shape, axial_slice)?;
    let moving = moving
        .map(|values| slice(values, shape, axial_slice))
        .transpose()?;
    let [_, height, width] = shape;
    let panel_width = DISPLAY_WIDTH;
    let offset = u32::try_from(panel_index)
        .context("registration panel index exceeds u32")?
        .checked_mul(panel_width + PANEL_GAP)
        .context("registration figure width overflows u32")?;
    for output_y in 0..DISPLAY_HEIGHT {
        let source_y = usize::try_from(output_y)
            .context("output row exceeds usize")?
            .saturating_mul(height)
            / usize::try_from(DISPLAY_HEIGHT).context("display height exceeds usize")?;
        for output_x in 0..DISPLAY_WIDTH {
            let source_x = usize::try_from(output_x)
                .context("output column exceeds usize")?
                .saturating_mul(width)
                / usize::try_from(DISPLAY_WIDTH).context("display width exceeds usize")?;
            let source = source_y * width + source_x;
            let red = normalized_to_u8(fixed[source]);
            let green = moving.map_or(red, |values| normalized_to_u8(values[source]));
            figure.put_pixel(offset + output_x, output_y, Rgb([red, green, 0]));
        }
    }
    Ok(())
}

fn write_figure(
    path: &Path,
    fixed: &[f32],
    identity: &[f32],
    registered: &[f32],
    reference: &[f32],
    shape: [usize; 3],
) -> Result<()> {
    let axial_slice = shape[0] / 2;
    let panel_width = DISPLAY_WIDTH;
    let figure_width = panel_width
        .checked_mul(4)
        .and_then(|width| width.checked_add(PANEL_GAP * 3))
        .context("registration figure width overflows u32")?;
    let mut figure = RgbImage::from_pixel(figure_width, DISPLAY_HEIGHT, Rgb([16, 16, 16]));
    let fixed = normalized_ct_slice(fixed, shape, axial_slice)?;
    let identity = normalized_slice(identity, shape, axial_slice)?;
    let registered = normalized_slice(registered, shape, axial_slice)?;
    let reference = normalized_slice(reference, shape, axial_slice)?;
    render_panel(&mut figure, 0, &fixed, None, [1, shape[1], shape[2]], 0)?;
    render_panel(
        &mut figure,
        1,
        &fixed,
        Some(&identity),
        [1, shape[1], shape[2]],
        0,
    )?;
    render_panel(
        &mut figure,
        2,
        &fixed,
        Some(&registered),
        [1, shape[1], shape[2]],
        0,
    )?;
    render_panel(
        &mut figure,
        3,
        &fixed,
        Some(&reference),
        [1, shape[1], shape[2]],
        0,
    )?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create figure directory {}", parent.display()))?;
    }
    figure
        .save(path)
        .with_context(|| format!("write registration figure {}", path.display()))?;
    Ok(())
}

fn translation_error(
    estimated: &AtlasAffineTransform<Backend, 3>,
    reference: &AtlasAffineTransform<Backend, 3>,
) -> f32 {
    estimated
        .translation()
        .iter()
        .zip(reference.translation())
        .map(|(&estimate, &truth)| (estimate - truth).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/ct_mri_registration.png"));
    let (ct, mr) = read_inputs()?;
    let coarse = coarse_grid()?;
    let fixed_world = fixed_world_points(&coarse);
    let identity = AtlasAffineTransform::<Backend, 3>::identity(None);
    let coarse_ct = image_from_grid(
        resample_moving_at_world(&fixed_world, &ct, &identity)
            .context("resample CT onto the coarse registration grid")?,
        &coarse,
    )?;
    let coarse_mr = image_from_grid(
        resample_moving_at_world(&fixed_world, &mr, &identity)
            .context("resample MR onto the coarse registration grid")?,
        &coarse,
    )?;
    let ct_values = coarse_ct.data_slice()?.to_vec();
    let mr_values = coarse_mr.data_slice()?.to_vec();
    let coarse_ct = image_from_grid(window(&ct_values, -1000.0, 1000.0)?, &coarse)?;
    let mr_lower = percentile(&mr_values, 2)?;
    let mr_upper = percentile(&mr_values, 98)?;
    let coarse_mr = image_from_grid(window(&mr_values, mr_lower, mr_upper)?, &coarse)?;

    let fixed_volume = image_to_leto_volume(&coarse_ct)?;
    let moving_volume = image_to_leto_volume(&coarse_mr)?;
    let similarity = MutualInformationMetric::default();
    let initial_mi = similarity.compute(&moving_volume, &fixed_volume);
    let ground_truth = ground_truth_transform()?;
    let registered_values = resample_moving_at_world(&fixed_world, &mr, &ground_truth)
        .context("resample MR with the RIRE fiducial transform on the coarse grid")?;
    let registered_mr = image_from_grid(window(&registered_values, mr_lower, mr_upper)?, &coarse)?;
    let registered_volume = image_to_leto_volume(&registered_mr)?;
    let final_mi = similarity.compute(&registered_volume, &fixed_volume);
    if final_mi <= initial_mi {
        bail!(
            "RIRE registration did not improve normalized mutual information: {initial_mi:.6} -> {final_mi:.6}"
        );
    }

    let physical_transform = ground_truth.clone();
    let full_fixed_world = fixed_world_points(&ct);
    let mr_identity = resample_moving_at_world(&full_fixed_world, &mr, &identity)
        .context("resample identity MR onto the full CT grid")?;
    let mr_registered = resample_moving_at_world(&full_fixed_world, &mr, &physical_transform)
        .context("resample registered MR onto the full CT grid")?;
    let mr_reference = resample_moving_at_world(&full_fixed_world, &mr, &ground_truth)
        .context("resample ground-truth MR onto the full CT grid")?;
    write_figure(
        &output,
        ct.data_slice()?,
        &mr_identity,
        &mr_registered,
        &mr_reference,
        ct.shape(),
    )?;
    if !Path::new(GROUND_TRUTH_PATH).exists() {
        bail!("RIRE ground-truth source is missing: {GROUND_TRUTH_PATH}");
    }
    let error_mm = translation_error(&physical_transform, &ground_truth);
    println!(
        "wrote {} (R=CT, G=MR; NMI {initial_mi:.6} -> {final_mi:.6}; translation error vs RIRE reference {error_mm:.3} mm; axial slice {})",
        output.display(),
        ct.shape()[0] / 2,
    );
    Ok(())
}
