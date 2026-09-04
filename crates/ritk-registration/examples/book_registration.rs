//! Generate the real CT/MR registration figure used by the RITK mdBook.
//!
//! The example loads the in-tree RIRE Patient 001 CT and MR T1 volumes,
//! evaluates the classical mutual-information metric on a coarse common grid,
//! and resamples the original MR volume onto the full CT grid with the
//! dataset's fiducial CT-to-MR transform. The rendered figure compares the
//! identity and registered overlays and shows the MR intensity change caused
//! by resampling, rather than repeating the same transform call as a second
//! duplicate reference panel.
#![expect(
    clippy::print_stdout,
    reason = "RITK-LINT-1: example/test diagnostic output"
)]

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use coeus_core::SequentialBackend;
use eunomia::CastFrom;
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use ritk_filter::resample::native::{fixed_world_points, resample_moving_at_world};
use ritk_image::Image;
use ritk_io::{format::metaimage::native::MetaImageReader, ImageReader};
use ritk_registration::classical::{
    engine::{HistogramEstimator, IntensityRange, MutualInformationMetric, NmiNormalization},
    image_to_leto_volume,
};
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::transform::affine::AtlasAffineTransform;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

type Backend = SequentialBackend;

const CT_PATH: &str = "test_data/registration/rire/training_001_ct.mha";
const MR_PATH: &str = "test_data/registration/rire/training_001_mr_T1.mha";
const GROUND_TRUTH_PATH: &str =
    "test_data/registration/rire/training_001_ct_to_mr_T1_ground_truth.tfm";
const COARSE_ORIGIN: [f64; 3] = [35.0, 40.0, 40.0];
const COARSE_EXTENT: [f64; 3] = [65.0, 240.0, 240.0];
const COARSE_SHAPE: [usize; 3] = [8, 64, 64];
const DISPLAY_SIDE: usize = 256;
const PANEL_WIDTH: u32 = 320;
const PANEL_HEIGHT: u32 = 330;
const PANEL_GAP: u32 = 16;

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

fn normalized_slice(
    values: &[f32],
    shape: [usize; 3],
    z: usize,
    lower: f32,
    upper: f32,
) -> Result<Vec<f32>> {
    window(slice(values, shape, z)?, lower, upper)
}

fn normalized_ct_slice(values: &[f32], shape: [usize; 3], z: usize) -> Result<Vec<f32>> {
    window(slice(values, shape, z)?, -1000.0, 1000.0)
}

fn absolute_difference(left: &[f32], right: &[f32]) -> Result<(Vec<f32>, f32, f32)> {
    if left.len() != right.len() {
        bail!(
            "registration comparison requires equal planes, got {} and {} samples",
            left.len(),
            right.len()
        );
    }
    if left.is_empty() {
        bail!("registration comparison cannot use an empty plane");
    }
    let difference = left
        .iter()
        .zip(right)
        .map(|(&left, &right)| (left - right).abs())
        .collect::<Vec<_>>();
    let maximum = difference
        .iter()
        .copied()
        .max_by(f32::total_cmp)
        .context("registration comparison has no maximum")?;
    let sample_count = u32::try_from(difference.len())
        .context("registration comparison sample count exceeds u32")?;
    let mean = difference.iter().sum::<f32>() / f32::cast_from(sample_count);
    Ok((difference, maximum, mean))
}

enum PanelContent<'a> {
    Ct(&'a [f32]),
    Overlay { fixed: &'a [f32], moving: &'a [f32] },
    Difference { values: &'a [f32], scale: f32 },
}

struct SvgPanel<'a> {
    offset_x: u32,
    offset_y: u32,
    title: &'a str,
    subtitle: &'a str,
    content: PanelContent<'a>,
}

fn channel(value: f32) -> u8 {
    u8::cast_from(value.clamp(0.0, 255.0).round())
}

fn panel_png_data_uri(content: &PanelContent<'_>, shape: [usize; 3]) -> Result<String> {
    let [_, height, width] = shape;
    let plane_size = height
        .checked_mul(width)
        .context("registration panel plane size overflows usize")?;
    let raster_capacity = DISPLAY_SIDE
        .checked_mul(DISPLAY_SIDE)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("registration panel raster size overflows usize")?;
    let mut raster = Vec::with_capacity(raster_capacity);
    for display_y in 0..DISPLAY_SIDE {
        let source_y = display_y * height / DISPLAY_SIDE;
        for display_x in 0..DISPLAY_SIDE {
            let source_x = display_x * width / DISPLAY_SIDE;
            let source = source_y
                .checked_mul(width)
                .and_then(|offset| offset.checked_add(source_x))
                .context("registration panel source index overflows usize")?;
            if source >= plane_size {
                bail!("registration panel source index exceeds its plane");
            }
            let (red, green, blue) = match content {
                PanelContent::Ct(values) => {
                    let value = *values
                        .get(source)
                        .context("CT panel does not match its declared shape")?;
                    (value, value, value)
                }
                PanelContent::Overlay { fixed, moving } => (
                    *fixed
                        .get(source)
                        .context("overlay CT panel does not match its declared shape")?,
                    *moving
                        .get(source)
                        .context("overlay MR panel does not match its declared shape")?,
                    0.0,
                ),
                PanelContent::Difference { values, scale } => {
                    let value = values
                        .get(source)
                        .copied()
                        .context("difference panel does not match its declared shape")?
                        / *scale
                        * 255.0;
                    (value, value, value)
                }
            };
            raster.extend_from_slice(&[channel(red), channel(green), channel(blue)]);
        }
    }
    let display_side = u32::try_from(DISPLAY_SIDE).context("display side exceeds u32")?;
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&raster, display_side, display_side, ColorType::Rgb8)
        .context("encode registration panel as PNG")?;
    Ok(STANDARD.encode(png))
}

fn draw_panel(svg: &mut String, panel: SvgPanel<'_>, shape: [usize; 3]) -> Result<()> {
    let image_offset_x = PANEL_WIDTH
        .checked_sub(u32::try_from(DISPLAY_SIDE).context("display side exceeds u32")?)
        .context("registration panel is narrower than its image")?
        / 2;
    let image_offset_y = 58_u32;
    let display_side = u32::try_from(DISPLAY_SIDE).context("display side exceeds u32")?;
    let encoded = panel_png_data_uri(&panel.content, shape)?;
    writeln!(
        svg,
        "<g transform=\"translate({}, {})\"><rect width=\"{}\" height=\"{}\" rx=\"8\" fill=\"#f8fafc\" stroke=\"#cbd5e1\"/>",
        panel.offset_x, panel.offset_y, PANEL_WIDTH, PANEL_HEIGHT
    )?;
    writeln!(
        svg,
        "<text x=\"{}\" y=\"24\" text-anchor=\"middle\" class=\"title\">{}</text>",
        PANEL_WIDTH / 2,
        panel.title
    )?;
    writeln!(
        svg,
        "<text x=\"{}\" y=\"43\" text-anchor=\"middle\" class=\"subtitle\">{}</text>",
        PANEL_WIDTH / 2,
        panel.subtitle
    )?;
    writeln!(
        svg,
        "<image x=\"{image_offset_x}\" y=\"{image_offset_y}\" width=\"{display_side}\" height=\"{display_side}\" href=\"data:image/png;base64,{encoded}\" image-rendering=\"auto\"/>"
    )?;
    svg.push_str("</g>\n");
    Ok(())
}

fn write_figure(
    path: &Path,
    fixed: &[f32],
    identity: &[f32],
    registered: &[f32],
    shape: [usize; 3],
    mr_lower: f32,
    mr_upper: f32,
) -> Result<(f32, f32)> {
    let axial_slice = shape[0] / 2;
    let fixed = normalized_ct_slice(fixed, shape, axial_slice)?;
    let identity = normalized_slice(identity, shape, axial_slice, mr_lower, mr_upper)?;
    let registered = normalized_slice(registered, shape, axial_slice, mr_lower, mr_upper)?;
    let (difference, maximum, mean) = absolute_difference(&identity, &registered)?;
    let figure_width = PANEL_WIDTH
        .checked_mul(2)
        .and_then(|width| width.checked_add(PANEL_GAP))
        .context("registration figure width overflows u32")?;
    let figure_height = PANEL_HEIGHT
        .checked_mul(2)
        .and_then(|height| height.checked_add(PANEL_GAP))
        .context("registration figure height overflows u32")?;
    let mut svg = String::from("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 ");
    writeln!(svg, "{figure_width} {figure_height}\">\n<style>.title{{font:600 16px sans-serif;fill:#172033}}.subtitle{{font:12px sans-serif;fill:#475569}}</style>")?;
    let plane_shape = [1, shape[1], shape[2]];
    draw_panel(
        &mut svg,
        SvgPanel {
            offset_x: 0,
            offset_y: 0,
            title: "CT reference",
            subtitle: "windowed to [-1000, 1000] HU",
            content: PanelContent::Ct(&fixed),
        },
        plane_shape,
    )?;
    draw_panel(
        &mut svg,
        SvgPanel {
            offset_x: PANEL_WIDTH + PANEL_GAP,
            offset_y: 0,
            title: "Identity overlay",
            subtitle: "R=CT, G=MR; no transform",
            content: PanelContent::Overlay {
                fixed: &fixed,
                moving: &identity,
            },
        },
        plane_shape,
    )?;
    draw_panel(
        &mut svg,
        SvgPanel {
            offset_x: 0,
            offset_y: PANEL_HEIGHT + PANEL_GAP,
            title: "Registered overlay",
            subtitle: "R=CT, G=MR; RIRE rigid transform",
            content: PanelContent::Overlay {
                fixed: &fixed,
                moving: &registered,
            },
        },
        plane_shape,
    )?;
    let difference_subtitle = format!("|registered - identity|; max {maximum:.2}, mean {mean:.2}");
    draw_panel(
        &mut svg,
        SvgPanel {
            offset_x: PANEL_WIDTH + PANEL_GAP,
            offset_y: PANEL_HEIGHT + PANEL_GAP,
            title: "MR resampling change",
            subtitle: &difference_subtitle,
            content: PanelContent::Difference {
                values: &difference,
                scale: maximum.max(f32::EPSILON),
            },
        },
        plane_shape,
    )?;
    svg.push_str("</svg>\n");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create figure directory {}", parent.display()))?;
    }
    std::fs::write(path, svg)
        .with_context(|| format!("write registration figure {}", path.display()))?;
    Ok((maximum, mean))
}

fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/ct_mri_registration.svg"));
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
    let coarse_ct = image_from_grid(window(coarse_ct.data_slice()?, -1000.0, 1000.0)?, &coarse)?;
    let mr_lower = percentile(coarse_mr.data_slice()?, 2)?;
    let mr_upper = percentile(coarse_mr.data_slice()?, 98)?;
    let coarse_mr = image_from_grid(
        window(coarse_mr.data_slice()?, mr_lower, mr_upper)?,
        &coarse,
    )?;

    let fixed_volume = image_to_leto_volume(&coarse_ct)?;
    let moving_volume = image_to_leto_volume(&coarse_mr)?;
    let display_range = IntensityRange::try_new(0.0, 255.0)?;
    let similarity = MutualInformationMetric::with_ranges(
        48,
        display_range,
        display_range,
        NmiNormalization::JointEntropy,
        HistogramEstimator::MovingLinearPartialVolume,
    )?;
    let initial_mi = similarity.compute(&fixed_volume, &moving_volume)?;
    let ground_truth = ground_truth_transform()?;
    let registered_values = resample_moving_at_world(&fixed_world, &mr, &ground_truth)
        .context("resample MR with the RIRE fiducial transform on the coarse grid")?;
    let registered_mr = image_from_grid(window(&registered_values, mr_lower, mr_upper)?, &coarse)?;
    let registered_volume = image_to_leto_volume(&registered_mr)?;
    let final_mi = similarity.compute(&fixed_volume, &registered_volume)?;
    if final_mi <= initial_mi {
        bail!(
            "RIRE registration did not improve normalized mutual information on the {:?} coarse grid with CT [-1000, 1000] and MR p2/p98 [{mr_lower:.3}, {mr_upper:.3}] windows: {initial_mi:.6} -> {final_mi:.6}",
            COARSE_SHAPE
        );
    }

    let full_fixed_world = fixed_world_points(&ct);
    let mr_identity = resample_moving_at_world(&full_fixed_world, &mr, &identity)
        .context("resample identity MR onto the full CT grid")?;
    let mr_registered = resample_moving_at_world(&full_fixed_world, &mr, &ground_truth)
        .context("resample registered MR onto the full CT grid")?;
    let (maximum_change, mean_change) = write_figure(
        &output,
        ct.data_slice()?,
        &mr_identity,
        &mr_registered,
        ct.shape(),
        mr_lower,
        mr_upper,
    )?;
    println!(
        "wrote {} (R=CT, G=MR; NMI {initial_mi:.6} -> {final_mi:.6}; MR change max {maximum_change:.2}, mean {mean_change:.2}; axial slice {})",
        output.display(),
        ct.shape()[0] / 2,
    );
    Ok(())
}
