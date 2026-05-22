//! Visualisation helpers for the RIRE CT/MRI T1 registration validation example.

use image::{Rgb, RgbImage};

/// Print a simple ASCII bar chart for an NCC value in `[-1, 1]`.
pub fn ncc_bar(label: &str, value: f64) {
    let width = 40_usize;
    // Map NCC ∈ [-1,1] → bar position in [0, width].
    let pos = ((value + 1.0) / 2.0 * width as f64)
        .round()
        .clamp(0.0, width as f64) as usize;
    let bar: String = (0..width)
        .map(|i| if i < pos { '█' } else { '░' })
        .collect();
    println!(" {:<28} [{bar}] {:.3}", label, value);
}

/// Save a 4-panel side-by-side PNG that shows pre-registration, the transform
/// effect, and post-registration states for a single axial slice.
///
/// # Panel layout (left → right)
///
/// | CT fixed (grey) | CT+MRI-pre overlay | Transform Δ | CT+MRI-post overlay |
///
/// **Overlay encoding**: red channel = CT (fixed), green channel = MRI (moving).
/// - Yellow / white pixels → anatomy overlap (R ≈ G) = good alignment.
/// - Red / green pixels → misalignment (R ≠ G) = registration needed.
///
/// **Transform Δ panel**: absolute per-voxel difference `|post − pre|`,
/// normalised to `[0, 255]`. Bright pixels show where the GT transform
/// moved tissue compared to the raw identity-mapped MRI.
///
/// Each panel is separated by a 4-pixel dark gap. A 20-pixel colour header
/// band identifies each panel (grey / red / yellow / green).
pub fn save_comparison_png(
    ct_slice: &[f32],   // [ny × nx] CT voxels (Hounsfield units)
    pre_slice: &[f32],  // [ny × nx] MRI without GT (identity resampled)
    post_slice: &[f32], // [ny × nx] MRI with GT alignment
    shape: [usize; 2],  // [ny, nx]
    output_path: &std::path::Path,
) -> anyhow::Result<()> {
    let [ny, nx] = shape;
    let gap: usize = 4; // pixels between panels
    let header: usize = 20; // coloured label band at top of each panel
    let total_w = (4 * nx + 3 * gap) as u32;
    let total_h = (header + ny) as u32;
    let mut img = RgbImage::new(total_w, total_h);

    // Dark background
    for p in img.pixels_mut() {
        *p = Rgb([18u8, 18, 18]);
    }

    // ── Normalise each modality ───────────────────────────────────────────────
    // CT: soft-tissue window [-200, 500] HU → [0, 255]
    let ct_u8: Vec<u8> = ct_slice
        .iter()
        .map(|&v| (((v.clamp(-200.0, 500.0) + 200.0) / 700.0) * 255.0) as u8)
        .collect();

    // MRI: min-max normalise to [0, 255]; zero-heavy slices get all-black
    let mri_to_u8 = |data: &[f32]| -> Vec<u8> {
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(1e-8);
        data.iter()
            .map(|&v| (((v - min) / range) * 255.0) as u8)
            .collect()
    };
    let pre_u8 = mri_to_u8(pre_slice);
    let post_u8 = mri_to_u8(post_slice);

    // Transform Δ: absolute difference |post_u8 - pre_u8| normalised to [0,255]
    let diff_raw: Vec<f32> = pre_u8
        .iter()
        .zip(post_u8.iter())
        .map(|(&a, &b)| (a as f32 - b as f32).abs())
        .collect();
    let diff_u8 = mri_to_u8(&diff_raw);

    // ── Helper: x offset for panel p ─────────────────────────────────────────
    let px_off = |p: usize| (p * (nx + gap)) as u32;

    // ── Panel 0: CT greyscale ─────────────────────────────────────────────────
    for iy in 0..ny {
        for ix in 0..nx {
            let v = ct_u8[iy * nx + ix];
            img.put_pixel(px_off(0) + ix as u32, (header + iy) as u32, Rgb([v, v, v]));
        }
    }

    // ── Panel 1: CT (red) + MRI-unaligned (green) overlay — PRE ──────────────
    for iy in 0..ny {
        for ix in 0..nx {
            let r = ct_u8[iy * nx + ix];
            let g = pre_u8[iy * nx + ix];
            img.put_pixel(px_off(1) + ix as u32, (header + iy) as u32, Rgb([r, g, 0]));
        }
    }

    // ── Panel 2: Transform Δ = |post − pre| greyscale ────────────────────────
    for iy in 0..ny {
        for ix in 0..nx {
            let v = diff_u8[iy * nx + ix];
            img.put_pixel(px_off(2) + ix as u32, (header + iy) as u32, Rgb([v, v, v]));
        }
    }

    // ── Panel 3: CT (red) + MRI-GT-aligned (green) overlay — POST ────────────
    for iy in 0..ny {
        for ix in 0..nx {
            let r = ct_u8[iy * nx + ix];
            let g = post_u8[iy * nx + ix];
            img.put_pixel(px_off(3) + ix as u32, (header + iy) as u32, Rgb([r, g, 0]));
        }
    }

    // ── Coloured header bands (identify each panel visually) ──────────────────
    // Grey = CT, Red = pre-overlay, Yellow = transform Δ, Green = post-overlay
    let header_colors: [Rgb<u8>; 4] = [
        Rgb([180, 180, 180]), // grey → CT (fixed)
        Rgb([220, 60, 60]),   // red → pre-registration
        Rgb([220, 200, 40]),  // yellow → transform Δ
        Rgb([60, 220, 60]),   // green → post-registration
    ];
    for (panel, &col) in header_colors.iter().enumerate() {
        for row in 0..header {
            for ix in 0..nx {
                img.put_pixel(px_off(panel) + ix as u32, row as u32, col);
            }
        }
    }

    img.save(output_path)
        .map_err(|e| anyhow::anyhow!("PNG save failed: {}", e))?;
    Ok(())
}
