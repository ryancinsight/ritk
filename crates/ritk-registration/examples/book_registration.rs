//! Generate the synthetic registration figure used by the RITK mdBook.
//!
//! The figure is produced from a deterministic translated phantom and the
//! public classic Thirion Demons implementation. The SVG writer is only a
//! presentation boundary; all image alignment values come from RITK.

use anyhow::{bail, Context, Result};
use ritk_filter::GaussianSigma;
use ritk_registration::demons::{DemonsConfig, ThirionDemonsRegistration};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const DIMS: [usize; 3] = [3, 64, 96];
const SHIFT_X: usize = 5;
const PANEL_WIDTH: u32 = 256;
const PANEL_HEIGHT: u32 = 288;

fn phantom() -> Result<Vec<f32>> {
    let [depth, height, width] = DIMS;
    let plane = height
        .checked_mul(width)
        .context("phantom plane overflows")?;
    let mut values =
        Vec::with_capacity(depth.checked_mul(plane).context("phantom size overflows")?);
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let x = f32::from(u16::try_from(x).context("x coordinate exceeds u16")?);
                let y = f32::from(u16::try_from(y).context("y coordinate exceeds u16")?);
                let z = f32::from(u16::try_from(z).context("z coordinate exceeds u16")?);
                let primary = ((x - 34.0).powi(2) / 15.0_f32.powi(2)
                    + (y - 33.0).powi(2) / 11.0_f32.powi(2))
                .sqrt();
                let secondary = ((x - 70.0).powi(2) + (y - 23.0).powi(2)).sqrt();
                let ring = if (primary - 1.0).abs() < 0.08 {
                    0.75
                } else {
                    0.0
                };
                let blob = 0.85 * (-0.5 * primary.powi(2)).exp();
                let satellite = 0.45 * (-(secondary / 8.0).powi(2)).exp();
                let depth_scale = 1.0 - 0.06 * z;
                values.push((depth_scale * (blob + satellite + ring)).clamp(0.0, 1.0));
            }
        }
    }
    Ok(values)
}

fn translate_x(fixed: &[f32]) -> Result<Vec<f32>> {
    let [depth, height, width] = DIMS;
    let plane = height.checked_mul(width).context("image plane overflows")?;
    let mut moving = vec![0.0; fixed.len()];
    for z in 0..depth {
        for y in 0..height {
            for x in SHIFT_X..width {
                let destination = z * plane + y * width + x;
                let source = z * plane + y * width + x - SHIFT_X;
                moving[destination] = fixed
                    .get(source)
                    .copied()
                    .context("translation source is out of bounds")?;
            }
        }
    }
    Ok(moving)
}

fn mse(left: &[f32], right: &[f32]) -> Result<f64> {
    if left.len() != right.len() || left.is_empty() {
        bail!("MSE requires equally sized, non-empty images")
    }
    let sum = left
        .iter()
        .zip(right)
        .map(|(&a, &b)| {
            let difference = f64::from(a) - f64::from(b);
            difference * difference
        })
        .sum::<f64>();
    let count = f64::from(u32::try_from(left.len()).context("voxel count exceeds u32")?);
    Ok(sum / count)
}

fn normalize(values: &[f32]) -> Result<Vec<f32>> {
    let maximum = values
        .iter()
        .copied()
        .fold(0.0_f32, f32::max)
        .max(f32::EPSILON);
    Ok(values
        .iter()
        .map(|value| (*value / maximum).clamp(0.0, 1.0))
        .collect())
}

fn svg_scalar_panel(svg: &mut String, values: &[f32], title: &str, offset_x: u32) -> Result<()> {
    let [_, height, width] = DIMS;
    let values = normalize(values)?;
    let width_u32 = u32::try_from(width).context("panel width exceeds u32")?;
    let height_u32 = u32::try_from(height).context("panel height exceeds u32")?;
    let cell_x = f64::from(PANEL_WIDTH - 32) / f64::from(width_u32);
    let cell_y = f64::from(PANEL_HEIGHT - 48) / f64::from(height_u32);
    writeln!(svg, "<g transform=\"translate({offset_x},0)\">")?;
    writeln!(
        svg,
        "<text x=\"{half}\" y=\"22\" text-anchor=\"middle\" class=\"title\">{title}</text>",
        half = PANEL_WIDTH / 2
    )?;
    let z = DIMS[0] / 2;
    let plane = height * width;
    for y in 0..height {
        for x in 0..width {
            let index = z * plane + y * width + x;
            let intensity =
                f64::from(values.get(index).copied().context("panel shape mismatch")?) * 255.0;
            let x0 = 16.0 + f64::from(u32::try_from(x).context("x index exceeds u32")?) * cell_x;
            let y0 = 32.0 + f64::from(u32::try_from(y).context("y index exceeds u32")?) * cell_y;
            writeln!(svg, "<rect x=\"{x0:.3}\" y=\"{y0:.3}\" width=\"{cell_x:.3}\" height=\"{cell_y:.3}\" fill=\"rgb({intensity:.0},{intensity:.0},{intensity:.0})\"/>")?;
        }
    }
    svg.push_str("</g>\n");
    Ok(())
}

fn svg_displacement_panel(svg: &mut String, displacement: &[f32], offset_x: u32) -> Result<()> {
    let [_, height, width] = DIMS;
    let maximum = displacement
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0, f32::max)
        .max(1e-6);
    let width_u32 = u32::try_from(width).context("panel width exceeds u32")?;
    let height_u32 = u32::try_from(height).context("panel height exceeds u32")?;
    let cell_x = f64::from(PANEL_WIDTH - 32) / f64::from(width_u32);
    let cell_y = f64::from(PANEL_HEIGHT - 48) / f64::from(height_u32);
    writeln!(svg, "<g transform=\"translate({offset_x},0)\">")?;
    writeln!(svg, "<text x=\"{half}\" y=\"22\" text-anchor=\"middle\" class=\"title\">Final x displacement</text>", half = PANEL_WIDTH / 2)?;
    let z = DIMS[0] / 2;
    let plane = height * width;
    for y in 0..height {
        for x in 0..width {
            let value = displacement
                .get(z * plane + y * width + x)
                .copied()
                .context("displacement shape mismatch")?;
            let positive = f64::from(value / maximum).clamp(-1.0, 1.0);
            let red = 32.0 + 223.0 * positive.max(0.0);
            let blue = 32.0 + 223.0 * (-positive).max(0.0);
            let x0 = 16.0 + f64::from(u32::try_from(x).context("x index exceeds u32")?) * cell_x;
            let y0 = 32.0 + f64::from(u32::try_from(y).context("y index exceeds u32")?) * cell_y;
            writeln!(svg, "<rect x=\"{x0:.3}\" y=\"{y0:.3}\" width=\"{cell_x:.3}\" height=\"{cell_y:.3}\" fill=\"rgb({red:.0},32,{blue:.0})\"/>")?;
        }
    }
    svg.push_str("</g>\n");
    Ok(())
}

fn write_figure(
    path: &Path,
    fixed: &[f32],
    moving: &[f32],
    warped: &[f32],
    displacement: &[f32],
) -> Result<()> {
    let figure_width = PANEL_WIDTH
        .checked_mul(4)
        .context("figure width overflows")?;
    let mut svg = String::from("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 ");
    writeln!(svg, "{figure_width} {PANEL_HEIGHT}\">\n<style>.title{{font:600 14px sans-serif;fill:#172033}}</style>")?;
    svg_scalar_panel(&mut svg, fixed, "Fixed", 0)?;
    svg_scalar_panel(&mut svg, moving, "Moving (+5 voxels)", PANEL_WIDTH)?;
    svg_scalar_panel(&mut svg, warped, "Warped moving", PANEL_WIDTH * 2)?;
    svg_displacement_panel(&mut svg, displacement, PANEL_WIDTH * 3)?;
    svg.push_str("</svg>\n");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create figure directory {}", parent.display()))?;
    }
    std::fs::write(path, svg).with_context(|| format!("write figure {}", path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/thirion_demons.svg"));
    let fixed = phantom()?;
    let moving = translate_x(&fixed)?;
    let initial_mse = mse(&fixed, &moving)?;
    let sigma = GaussianSigma::new(1.0).context("diffusion sigma must be positive")?;
    let registration = ThirionDemonsRegistration::new(DemonsConfig {
        max_iterations: 30,
        sigma_diffusion: Some(sigma),
        sigma_fluid: None,
        max_step_length: 2.0,
    });
    let result = registration.register(&fixed, &moving, DIMS, [1.0, 1.0, 1.0])?;
    if result.final_mse >= initial_mse {
        bail!(
            "registration did not improve MSE: initial={initial_mse:.6}, final={:.6}",
            result.final_mse
        )
    }
    write_figure(&output, &fixed, &moving, &result.warped, &result.disp_x)?;
    println!(
        "wrote {} (MSE {initial_mse:.6} -> {:.6}, iterations={})",
        output.display(),
        result.final_mse,
        result.num_iterations
    );
    Ok(())
}
