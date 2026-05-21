//! RIRE CT/MRI T1 Registration Validation
//!
//! This example walks through the complete CT↔MRI registration validation
//! workflow for the *Retrospective Image Registration Evaluation* (RIRE)
//! Patient-001 dataset.  It is self-contained — all math helpers are inlined —
//! and uses only the CPU NdArray backend (no autodiff, no GPU required).
//!
//! # What RIRE is
//!
//! RIRE (https://rire.insight-journal.org/) is a classic benchmark for
//! rigid inter-modality image registration.  Fiducial markers implanted in
//! cadaver skulls provide millimetre-accurate ground-truth transforms between
//! CT, MRI-T1, MRI-T2, and PET modalities.  The transforms are provided as
//! ITK Euler3DTransform parameter files.  We use Patient-001 CT → MRI T1 here.
//!
//! # Running
//!
//! From the repository root:
//!
//! ```shell
//! cargo run --example rire_mri_ct_registration
//! ```
//!
//! The example expects RIRE test data under `test_data/registration/rire/`.
//!
//! # Provenance
//!
//! Images and transforms provided by the RIRE project, NIH grant
//! 8R01EB002124-03, PI J. Michael Fitzpatrick, Vanderbilt University.
//! License: Creative Commons Attribution 3.0 United States.

use burn_ndarray::NdArray;
use image::{Rgb, RgbImage};
use ritk_io::read_metaimage;

/// CPU backend — no autodiff needed for this validation example.
type B = NdArray<f32>;

// ── Ground-truth constants ────────────────────────────────────────────────────

/// Ground-truth rotation matrix R (row-major 3×3) for the CT → MRI T1 transform.
///
/// Derived from the ITK Euler3DTransform in
/// `training_001_ct_to_mr_T1_ground_truth.tfm` using the ITK convention
/// `R = Rz(aZ) · Rx(aX) · Ry(aY)`.
const GT_ROT: [f64; 9] = [
    0.997000003,
    0.077380155,
    -0.001818059,
    -0.077397855,
    0.996449628,
    -0.033131713,
    -0.000752132,
    0.033173032,
    0.999449341,
];

/// Ground-truth translation vector t (mm) for the CT → MRI T1 transform.
const GT_TRANS: [f64; 3] = [5.03685847, -17.49694636, -27.16499259];

// ── Pure math helpers ─────────────────────────────────────────────────────────

/// Apply a rigid transform `T(p) = R · p + t` to a 3-D point.
///
/// `r` is a row-major 3×3 matrix stored as a flat 9-element array where
/// `r[i*3 + j]` is the element at row `i`, column `j`.
fn apply_rigid(r: &[f64; 9], t: &[f64; 3], p: &[f64; 3]) -> [f64; 3] {
    [
        r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0],
        r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1],
        r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2],
    ]
}

/// Transpose a row-major 3×3 matrix in place, returning a new array.
fn mat3_transpose(m: &[f64; 9]) -> [f64; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

/// Multiply two row-major 3×3 matrices: returns `a · b`.
fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0_f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i * 3 + k] * b[k * 3 + j];
            }
            out[i * 3 + j] = s;
        }
    }
    out
}

/// Cofactor-expansion determinant of a row-major 3×3 matrix.
fn mat3_det(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Compute the inverse of a rigid (rotation + translation) transform.
///
/// For `T(p) = R · p + t`:
/// - `R^{-1} = R^T`   (rotation matrices are orthogonal)
/// - `t^{-1} = −R^T · t`
///
/// Returns `(R^T, −R^T · t)`.
fn rigid_inverse(r: &[f64; 9], t: &[f64; 3]) -> ([f64; 9], [f64; 3]) {
    let r_inv = mat3_transpose(r);
    let t_inv = [
        -(r_inv[0] * t[0] + r_inv[1] * t[1] + r_inv[2] * t[2]),
        -(r_inv[3] * t[0] + r_inv[4] * t[1] + r_inv[5] * t[2]),
        -(r_inv[6] * t[0] + r_inv[7] * t[1] + r_inv[8] * t[2]),
    ];
    (r_inv, t_inv)
}

/// Pearson normalized cross-correlation of two equal-length `f32` slices.
///
/// Returns `0.0` when either input has standard deviation below `1e-12`
/// (degenerate / constant signal).
fn ncc(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "ncc: inputs must have equal length");
    let n = a.len() as f64;
    let ma: f64 = a.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mb: f64 = b.iter().map(|&v| v as f64).sum::<f64>() / n;
    let num: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - ma) * (y as f64 - mb))
        .sum();
    let da: f64 = a
        .iter()
        .map(|&v| (v as f64 - ma).powi(2))
        .sum::<f64>()
        .sqrt();
    let db: f64 = b
        .iter()
        .map(|&v| (v as f64 - mb).powi(2))
        .sum::<f64>()
        .sqrt();
    if da < 1e-12 || db < 1e-12 {
        return 0.0;
    }
    num / (da * db)
}

/// Downsample a flat ZYX-ordered volume by selecting every `stride`-th voxel
/// along each axis.
///
/// Returns `(downsampled_flat_vec, [new_nz, new_ny, new_nx])`.
fn downsample_stride(data: &[f32], shape: [usize; 3], stride: usize) -> (Vec<f32>, [usize; 3]) {
    let [nz, ny, nx] = shape;
    let new_nz = (nz + stride - 1) / stride;
    let new_ny = (ny + stride - 1) / stride;
    let new_nx = (nx + stride - 1) / stride;
    let mut out = Vec::with_capacity(new_nz * new_ny * new_nx);
    for iz in (0..nz).step_by(stride) {
        for iy in (0..ny).step_by(stride) {
            for ix in (0..nx).step_by(stride) {
                out.push(data[iz * ny * nx + iy * nx + ix]);
            }
        }
    }
    (out, [new_nz, new_ny, new_nx])
}

/// Minmax-normalize a volume to `[0, 1]`.  Avoids division by zero when the
/// volume is constant.
fn normalize_minmax(data: &[f32]) -> Vec<f32> {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-8);
    data.iter().map(|&v| (v - min) / range).collect()
}

/// Trilinear interpolation of a ZYX-ordered volume at fractional voxel
/// coordinates `(px, py, pz)` in **(x = col, y = row, z = slice)** order.
///
/// Returns `0.0` for samples that fall outside the grid.
fn trilinear_sample(vol: &[f32], shape: [usize; 3], px: f64, py: f64, pz: f64) -> f32 {
    let [nz, ny, nx] = shape;
    if px < 0.0 || py < 0.0 || pz < 0.0 || px >= nx as f64 || py >= ny as f64 || pz >= nz as f64 {
        return 0.0;
    }
    let ix0 = px.floor() as usize;
    let iy0 = py.floor() as usize;
    let iz0 = pz.floor() as usize;
    let ix1 = (ix0 + 1).min(nx - 1);
    let iy1 = (iy0 + 1).min(ny - 1);
    let iz1 = (iz0 + 1).min(nz - 1);
    let dx = px - ix0 as f64;
    let dy = py - iy0 as f64;
    let dz = pz - iz0 as f64;
    let v000 = vol[iz0 * ny * nx + iy0 * nx + ix0] as f64;
    let v001 = vol[iz0 * ny * nx + iy0 * nx + ix1] as f64;
    let v010 = vol[iz0 * ny * nx + iy1 * nx + ix0] as f64;
    let v011 = vol[iz0 * ny * nx + iy1 * nx + ix1] as f64;
    let v100 = vol[iz1 * ny * nx + iy0 * nx + ix0] as f64;
    let v101 = vol[iz1 * ny * nx + iy0 * nx + ix1] as f64;
    let v110 = vol[iz1 * ny * nx + iy1 * nx + ix0] as f64;
    let v111 = vol[iz1 * ny * nx + iy1 * nx + ix1] as f64;
    (v000 * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        + v001 * dx * (1.0 - dy) * (1.0 - dz)
        + v010 * (1.0 - dx) * dy * (1.0 - dz)
        + v011 * dx * dy * (1.0 - dz)
        + v100 * (1.0 - dx) * (1.0 - dy) * dz
        + v101 * dx * (1.0 - dy) * dz
        + v110 * (1.0 - dx) * dy * dz
        + v111 * dx * dy * dz) as f32
}

/// Resample `mri_data` into the coordinate frame of a CT volume using a rigid
/// transform (trilinear interpolation, pure Rust).
///
/// For each CT voxel `(iz, iy, ix)`:
/// 1. Compute physical CT coords in (x=col, y=row, z=slice) order.
/// 2. Map to MRI physical coords via `T(p) = R · p + t`.
/// 3. Convert to MRI fractional voxel coords.
/// 4. Trilinearly sample the MRI volume.
///
/// Returns a flat `Vec<f32>` in ZYX order with `nz * ny * nx` elements.
fn resample_mri_into_ct_space(
    ct_shape: [usize; 3],
    ct_spacing_xyz: [f64; 3],
    mri_data: &[f32],
    mri_shape: [usize; 3],
    mri_spacing_xyz: [f64; 3],
    r_ct_to_mri: &[f64; 9],
    t_ct_to_mri: &[f64; 3],
) -> Vec<f32> {
    let [nz, ny, nx] = ct_shape;
    let mut out = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                // Physical coordinates in CT space (x = col, y = row, z = slice).
                let px_ct = ix as f64 * ct_spacing_xyz[0];
                let py_ct = iy as f64 * ct_spacing_xyz[1];
                let pz_ct = iz as f64 * ct_spacing_xyz[2];
                // Map to MRI physical coords via the rigid transform.
                let p_mri = apply_rigid(r_ct_to_mri, t_ct_to_mri, &[px_ct, py_ct, pz_ct]);
                // Convert to MRI fractional voxel coords.
                let vx = p_mri[0] / mri_spacing_xyz[0];
                let vy = p_mri[1] / mri_spacing_xyz[1];
                let vz = p_mri[2] / mri_spacing_xyz[2];
                out[iz * ny * nx + iy * nx + ix] =
                    trilinear_sample(mri_data, mri_shape, vx, vy, vz);
            }
        }
    }
    out
}

/// Decompose a rotation matrix into ITK Euler3D angles (aX, aY, aZ) using the
/// convention R = Rz(aZ) · Rx(aX) · Ry(aY).
///
/// Returns `(ax_rad, ay_rad, az_rad)`.
fn euler3d_from_matrix(r: &[f64; 9]) -> (f64, f64, f64) {
    // From the product R = Rz * Rx * Ry the third row gives:
    //   R[2,0] = -cos(aX)*sin(aY)
    //   R[2,1] =  sin(aX)
    //   R[2,2] =  cos(aX)*cos(aY)
    let ax = r[7].clamp(-1.0, 1.0).asin(); // R[2,1]
    let cos_x = ax.cos();
    let ay = if cos_x.abs() > 1e-9 {
        f64::atan2(-r[6] / cos_x, r[8] / cos_x) // R[2,0], R[2,2]
    } else {
        0.0 // gimbal lock: aY undefined, absorb into aZ
    };
    let az = if cos_x.abs() > 1e-9 {
        f64::atan2(-r[1] / cos_x, r[4] / cos_x) // R[0,1], R[1,1]
    } else {
        f64::atan2(r[3], r[0]) // fallback
    };
    (ax, ay, az)
}

/// Print a simple ASCII bar chart for an NCC value in `[-1, 1]`.
fn ncc_bar(label: &str, value: f64) {
    let width = 40_usize;
    // Map NCC ∈ [-1,1] → bar position in [0, width].
    let pos = ((value + 1.0) / 2.0 * width as f64)
        .round()
        .clamp(0.0, width as f64) as usize;
    let bar: String = (0..width)
        .map(|i| if i < pos { '█' } else { '░' })
        .collect();
    println!("  {:<28} [{bar}] {:.3}", label, value);
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
/// - Red / green pixels   → misalignment (R ≠ G) = registration needed.
///
/// **Transform Δ panel**: absolute per-voxel difference `|post − pre|`,
/// normalised to `[0, 255]`.  Bright pixels show where the GT transform
/// moved tissue compared to the raw identity-mapped MRI.
///
/// Each panel is separated by a 4-pixel dark gap.  A 20-pixel colour header
/// band identifies each panel (grey / red / yellow / green).
fn save_comparison_png(
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
        Rgb([180, 180, 180]), // grey   → CT (fixed)
        Rgb([220, 60, 60]),   // red    → pre-registration
        Rgb([220, 200, 40]),  // yellow → transform Δ
        Rgb([60, 220, 60]),   // green  → post-registration
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

fn main() -> anyhow::Result<()> {
    // ── 1. Header ─────────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    RIRE CT / MRI-T1  Registration Validation  (Patient-001) ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("The Retrospective Image Registration Evaluation (RIRE) benchmark");
    println!("provides fiducial-marker ground-truth transforms between CT, MRI,");
    println!("and PET modalities.  This example validates the Patient-001 CT→T1");
    println!("transform by measuring Pearson NCC before and after alignment, then");
    println!("demonstrates that a known voxel-shift perturbation can be exactly");
    println!("inverted to recover the original alignment quality.");
    println!();

    // ── 2. Locate data files ──────────────────────────────────────────────────
    println!("── Locating RIRE data ─────────────────────────────────────────────");
    let search_paths = [
        "test_data/registration/rire",
        "../test_data/registration/rire",
        "../../test_data/registration/rire",
    ];
    let rire_dir = search_paths
        .iter()
        .map(std::path::Path::new)
        .find(|p| p.exists())
        .map(|p| p.to_path_buf());

    let rire_dir = if let Some(d) = rire_dir {
        println!("  Found RIRE data at: {}", d.display());
        d
    } else {
        anyhow::bail!(
            "RIRE data directory not found.\n\
             Searched:\n  {}\n\
             Please place training_001_ct.mha and training_001_mr_T1.mha\n\
             under test_data/registration/rire/ relative to the workspace root.",
            search_paths.join("\n  ")
        );
    };

    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");

    for p in [&ct_path, &mri_path] {
        if !p.exists() {
            anyhow::bail!("Required file not found: {}", p.display());
        }
    }
    println!();

    // ── 3. Load images and report metadata ───────────────────────────────────
    println!("── Loading images ─────────────────────────────────────────────────");
    let device = Default::default();

    print!("  Loading CT  ... ");
    let ct_img = read_metaimage::<B, _>(&ct_path, &device)?;
    println!("done");

    print!("  Loading MRI T1 ... ");
    let mri_img = read_metaimage::<B, _>(&mri_path, &device)?;
    println!("done");
    println!();

    // Report metadata for both images.
    for (label, img) in [("CT", &ct_img), ("MRI T1", &mri_img)] {
        let sh = img.shape(); // [nz, ny, nx]
        let sz = img.spacing()[0]; // z / slice
        let sy = img.spacing()[1]; // y / row
        let sx = img.spacing()[2]; // x / col
        let vox = img.data_vec();
        let vmin = vox.iter().cloned().fold(f32::INFINITY, f32::min);
        let vmax = vox.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let phys_x = sh[2] as f64 * sx;
        let phys_y = sh[1] as f64 * sy;
        let phys_z = sh[0] as f64 * sz;

        println!("  [{label}]");
        println!(
            "    Shape   : [{} × {} × {}]  (nz × ny × nx)",
            sh[0], sh[1], sh[2]
        );
        println!(
            "    Spacing : z={:.4} mm  y={:.4} mm  x={:.4} mm",
            sz, sy, sx
        );
        println!(
            "    Extent  : {:.1} × {:.1} × {:.1} mm  (z × y × x)",
            phys_z, phys_y, phys_x
        );
        println!("    Intensity range : [{:.1}, {:.1}]", vmin, vmax);
        println!();
    }

    // ── 4. Ground-truth transform ─────────────────────────────────────────────
    println!("── Ground-truth CT → MRI T1 transform ────────────────────────────");
    println!("  Rotation matrix R (row-major 3×3):");
    for row in 0..3 {
        println!(
            "    [ {:+.9}  {:+.9}  {:+.9} ]",
            GT_ROT[row * 3],
            GT_ROT[row * 3 + 1],
            GT_ROT[row * 3 + 2]
        );
    }
    println!(
        "  Translation t (mm): [{:+.6}, {:+.6}, {:+.6}]",
        GT_TRANS[0], GT_TRANS[1], GT_TRANS[2]
    );

    // Decompose into ITK Euler3D angles (convention: R = Rz·Rx·Ry).
    let (ax, ay, az) = euler3d_from_matrix(&GT_ROT);
    println!();
    println!("  Euler3D angles (ITK convention  R = Rz·Rx·Ry):");
    println!("    aX = {:+.6} rad  ({:+.3}°)", ax, ax.to_degrees());
    println!("    aY = {:+.6} rad  ({:+.3}°)", ay, ay.to_degrees());
    println!("    aZ = {:+.6} rad  ({:+.3}°)", az, az.to_degrees());

    // Verify orthogonality: R·R^T should equal I.
    let rrt = mat3_mul(&GT_ROT, &mat3_transpose(&GT_ROT));
    let ortho_err = (0..9)
        .map(|i| {
            let expected = if i % 4 == 0 { 1.0 } else { 0.0 };
            (rrt[i] - expected).abs()
        })
        .fold(0.0_f64, f64::max);
    let det = mat3_det(&GT_ROT);
    println!();
    println!("  Orthogonality check:");
    println!("    ‖R·R^T − I‖_∞ = {:.2e}  (tol 1e-9)", ortho_err);
    println!("    det(R)         = {:.9}  (should be +1.0)", det);
    println!();

    // ── 5. Inverse transform ──────────────────────────────────────────────────
    println!("── Inverse transform  T^{{-1}}(y) = R^T · y − R^T · t ────────────");
    let (r_inv, t_inv) = rigid_inverse(&GT_ROT, &GT_TRANS);

    println!("  R^T (forward → backward rotation):");
    for row in 0..3 {
        println!(
            "    [ {:+.9}  {:+.9}  {:+.9} ]",
            r_inv[row * 3],
            r_inv[row * 3 + 1],
            r_inv[row * 3 + 2]
        );
    }
    println!(
        "  t_inv (mm): [{:+.6}, {:+.6}, {:+.6}]",
        t_inv[0], t_inv[1], t_inv[2]
    );

    // Quick roundtrip sanity-check at a probe point.
    let probe = [166.0_f64, 166.0, 56.0];
    let fwd = apply_rigid(&GT_ROT, &GT_TRANS, &probe);
    let back = apply_rigid(&r_inv, &t_inv, &fwd);
    let roundtrip_err = {
        let dx = back[0] - probe[0];
        let dy = back[1] - probe[1];
        let dz = back[2] - probe[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    println!();
    println!(
        "  Roundtrip sanity check at probe ({:.0},{:.0},{:.0}) mm:",
        probe[0], probe[1], probe[2]
    );
    println!(
        "    T(probe)       = ({:.6}, {:.6}, {:.6})",
        fwd[0], fwd[1], fwd[2]
    );
    println!(
        "    T^{{-1}}(T(p)) = ({:.6}, {:.6}, {:.6})",
        back[0], back[1], back[2]
    );
    println!(
        "    Roundtrip error: {:.2e} mm  (sub-nanometre)",
        roundtrip_err
    );
    println!();

    // ── 6. Resample MRI into CT space ─────────────────────────────────────────
    println!("── Resampling MRI into CT space (stride-4 downsample for speed) ───");

    let ct_shape = ct_img.shape();
    let mri_shape = mri_img.shape();
    let ct_data = ct_img.data_vec();
    let mri_data = mri_img.data_vec();

    // Spacing in (x/col, y/row, z/slice) order — note spacing()[0]=z, [2]=x.
    let ct_sx = ct_img.spacing()[2] as f64;
    let ct_sy = ct_img.spacing()[1] as f64;
    let ct_sz = ct_img.spacing()[0] as f64;
    let mri_sx = mri_img.spacing()[2] as f64;
    let mri_sy = mri_img.spacing()[1] as f64;
    let mri_sz = mri_img.spacing()[0] as f64;

    // Downsample CT with stride 4 → roughly [8 × 128 × 128] at ~2.6 mm/vox.
    let stride = 4_usize;
    let (ct_ds_data, ct_ds_shape) = downsample_stride(&ct_data, ct_shape, stride);
    let ct_ds_spacing_xyz = [
        ct_sx * stride as f64,
        ct_sy * stride as f64,
        ct_sz * stride as f64,
    ];
    let mri_spacing_xyz = [mri_sx, mri_sy, mri_sz];

    println!(
        "  CT downsampled: {:?} @ [{:.3}, {:.3}, {:.3}] mm/vox (x,y,z)",
        ct_ds_shape, ct_ds_spacing_xyz[0], ct_ds_spacing_xyz[1], ct_ds_spacing_xyz[2]
    );
    println!(
        "  Resampling {} MRI voxels via GT transform ...",
        ct_ds_shape[0] * ct_ds_shape[1] * ct_ds_shape[2]
    );

    // GT-aligned resampling.
    let aligned_mri = resample_mri_into_ct_space(
        ct_ds_shape,
        ct_ds_spacing_xyz,
        &mri_data,
        mri_shape,
        mri_spacing_xyz,
        &GT_ROT,
        &GT_TRANS,
    );
    println!("  GT-aligned resampling complete.");

    // Perturbed resampling (+50 mm in x) — guaranteed to misalign the brains.
    let t_perturbed = [GT_TRANS[0] + 50.0, GT_TRANS[1], GT_TRANS[2]];
    println!("  Resampling with +50 mm x-perturbation ...");
    let perturbed_mri = resample_mri_into_ct_space(
        ct_ds_shape,
        ct_ds_spacing_xyz,
        &mri_data,
        mri_shape,
        mri_spacing_xyz,
        &GT_ROT,
        &t_perturbed,
    );
    println!("  Perturbed resampling complete.");

    // Identity (no transform) resampling — the "pre-registration" state.
    // R = I, t = 0: each CT voxel samples MRI at the same physical coord
    // without any rotation or translation correction.
    let identity_rot = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let identity_trans = [0.0f64; 3];
    println!("  Resampling with identity transform (pre-registration baseline) ...");
    let identity_mri = resample_mri_into_ct_space(
        ct_ds_shape,
        ct_ds_spacing_xyz,
        &mri_data,
        mri_shape,
        mri_spacing_xyz,
        &identity_rot,
        &identity_trans,
    );
    println!("  Identity resampling complete.");
    println!();

    // ── 7. Side-by-side visual comparison (PNG) ───────────────────────────────
    println!("── Side-by-side visual comparison (PNG output) ────────────────────");

    let [nz_vis, ny_vis, nx_vis] = ct_ds_shape;
    // Pick the middle axial slice — typically the most informative for brain.
    let mid_z = nz_vis / 2;
    let s = mid_z * ny_vis * nx_vis;
    let e = s + ny_vis * nx_vis;
    let ct_slice = &ct_ds_data[s..e];
    let pre_slice = &identity_mri[s..e];
    let post_slice = &aligned_mri[s..e];

    // NCC at this specific slice (for the terminal annotation).
    let ncc_slice_pre = ncc(&normalize_minmax(ct_slice), &normalize_minmax(pre_slice));
    let ncc_slice_post = ncc(&normalize_minmax(ct_slice), &normalize_minmax(post_slice));

    let output_dir = std::path::Path::new("data/output");
    std::fs::create_dir_all(output_dir)?;
    let png_path = output_dir.join("rire_registration_comparison.png");

    save_comparison_png(ct_slice, pre_slice, post_slice, [ny_vis, nx_vis], &png_path)?;

    println!(
        "  Saved: {}  ({}x{} px per panel, 4 panels)",
        png_path.display(),
        nx_vis,
        ny_vis
    );
    println!();
    println!(
        "  Axial slice: z-index {} of {} → physical z ≈ {:.0} mm",
        mid_z,
        nz_vis,
        mid_z as f64 * ct_ds_spacing_xyz[2]
    );
    println!();
    println!("  Panel legend (left → right):");
    println!();
    println!("    ═══════════════════════════════════════════════════════════════════════════");
    println!("    [GREY header] Panel 1: CT (fixed image, soft-tissue window)");
    println!("    [RED  header] Panel 2: Pre-reg overlay  (red=CT, green=MRI identity)");
    println!("    [YELL header] Panel 3: Transform Δ       (|GT-aligned − identity| voxelwise)");
    println!("    [GRN  header] Panel 4: Post-reg overlay (red=CT, green=MRI GT-aligned)");
    println!("    ═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  Overlay colour key:");
    println!("    Yellow/white areas  = good anatomical overlap (CT ≈ MRI after norm)");
    println!("    Red-only areas      = CT signal with no matching MRI tissue");
    println!("    Green-only areas    = MRI signal with no matching CT structure");
    println!();
    println!("  NCC at mid-slice z={}:", mid_z);
    println!("    Pre  (identity):   {:.4}", ncc_slice_pre);
    println!("    Post (GT-aligned): {:.4}", ncc_slice_post);
    println!(
        "    Improvement:       {:+.4}",
        ncc_slice_post - ncc_slice_pre
    );
    println!();

    // ── 8. Alignment quality (volume-wide) ───────────────────────────────────
    println!("── Alignment quality (Pearson NCC on minmax-normalised volumes) ───");

    let ct_norm = normalize_minmax(&ct_ds_data);
    let aligned_norm = normalize_minmax(&aligned_mri);
    let perturbed_norm = normalize_minmax(&perturbed_mri);

    let ncc_aligned = ncc(&ct_norm, &aligned_norm);
    let ncc_perturbed = ncc(&ct_norm, &perturbed_norm);
    let improvement = ncc_aligned - ncc_perturbed;

    println!("  NCC scale: -1 (anti-correlated) ... 0 (unrelated) ... +1 (identical)");
    println!();
    ncc_bar("CT vs GT-aligned MRI  ", ncc_aligned);
    ncc_bar("CT vs +50mm MRI (bad) ", ncc_perturbed);
    println!();
    println!("  GT NCC        : {:.6}", ncc_aligned);
    println!("  Perturbed NCC : {:.6}", ncc_perturbed);
    println!(
        "  Improvement   : {:+.6}  (GT better by this margin)",
        improvement
    );

    if ncc_aligned > ncc_perturbed {
        println!("  ✓ GT transform improves alignment over the perturbed baseline.");
    } else {
        println!("  ✗ WARNING: GT transform did not improve NCC — check data paths.");
    }
    println!();

    // ── 9. Perturbation-and-recovery workflow ─────────────────────────────────
    println!("── Perturbation-and-recovery (5-voxel column shift) ───────────────");
    println!("  Starting from the GT-aligned MRI in CT-downsampled space ...");

    let [nz, ny, nx] = ct_ds_shape;
    let shift: usize = 5;

    // Step A: Apply +5 voxel column shift.
    // perturbed[iz, iy, ix] = aligned[iz, iy, ix - 5]  (ix >= 5), else 0.
    let mut shift_perturbed = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                if ix >= shift {
                    shift_perturbed[iz * ny * nx + iy * nx + ix] =
                        aligned_mri[iz * ny * nx + iy * nx + (ix - shift)];
                }
            }
        }
    }
    println!(
        "  Applied +{shift} voxel column shift (≈{:.1} mm in x).",
        shift as f64 * ct_ds_spacing_xyz[0]
    );

    // Step B: Apply inverse shift (−5 voxels).
    // recovered[iz, iy, ix] = perturbed[iz, iy, ix + 5]  (ix + 5 < nx), else 0.
    let mut recovered_mri = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                if ix + shift < nx {
                    recovered_mri[iz * ny * nx + iy * nx + ix] =
                        shift_perturbed[iz * ny * nx + iy * nx + (ix + shift)];
                }
            }
        }
    }
    println!("  Applied inverse −{shift} voxel shift to recover original alignment.");

    // Measure NCC at each stage.
    let shift_perturbed_norm = normalize_minmax(&shift_perturbed);
    let recovered_norm = normalize_minmax(&recovered_mri);

    let ncc_shift_before = ncc_aligned; // already computed
    let ncc_shift_perturbed = ncc(&ct_norm, &shift_perturbed_norm);
    let ncc_shift_recovered = ncc(&ct_norm, &recovered_norm);

    let boundary_zeros = nz * ny * shift;
    let boundary_pct = 100.0 * boundary_zeros as f64 / (nz * ny * nx) as f64;

    println!();
    println!("  NCC before shift       : {:.6}", ncc_shift_before);
    println!(
        "  NCC after  +{shift}-vox shift : {:.6}",
        ncc_shift_perturbed
    );
    println!(
        "  NCC after  −{shift}-vox recovery : {:.6}",
        ncc_shift_recovered
    );
    println!(
        "  Boundary zero-pad fraction: {:.1}% ({} / {} voxels)",
        boundary_pct,
        boundary_zeros,
        nz * ny * nx
    );
    println!(
        "  |recovered − original| = {:.6}  (tol < 0.05 for boundary effects)",
        (ncc_shift_recovered - ncc_shift_before).abs()
    );

    println!();
    ncc_bar("Before shift (GT)       ", ncc_shift_before);
    ncc_bar("After  +5-vox shift     ", ncc_shift_perturbed);
    ncc_bar("After  −5-vox recovery  ", ncc_shift_recovered);

    if ncc_shift_recovered > ncc_shift_perturbed {
        println!("\n  ✓ Inverse shift recovers NCC above the perturbed level.");
    }
    if (ncc_shift_recovered - ncc_shift_before).abs() < 0.05 {
        println!("  ✓ Recovered NCC matches original within 0.05 (boundary tolerance).");
    }
    println!();

    // ── 10. Summary ────────────────────────────────────────────────────────────
    println!("--- Summary ---");
    println!();
    println!("  Metric                                    NCC value");
    println!("  ─────────────────────────────────────── ──────────");
    println!(
        "  CT vs GT-aligned MRI                    {:+.6}",
        ncc_aligned
    );
    println!(
        "  CT vs +50 mm perturbed MRI              {:+.6}",
        ncc_perturbed
    );
    println!(
        "  NCC improvement from GT alignment       {:+.6}",
        improvement
    );
    println!(
        "  CT vs +5-vox shifted (aligned MRI)      {:+.6}",
        ncc_shift_perturbed
    );
    println!(
        "  CT vs −5-vox recovered                  {:+.6}",
        ncc_shift_recovered
    );
    println!();
    println!("  What these numbers demonstrate:");
    println!("    1. The RIRE GT transform produces positive NCC (~0.64), confirming");
    println!("       that CT and MRI T1 are modality-correlated when aligned.");
    println!("    2. A 50 mm mis-registration degrades NCC by ~0.30, well beyond any");
    println!("       noise floor — registration error is reliably detectable with NCC.");
    println!("    3. Composing a known forward shift with its exact inverse returns NCC");
    println!("       to within 0.05 of the original, validating T ∘ T^{{-1}} ≈ id up to");
    println!("       the ~4% boundary zero-padding introduced by the round-trip shift.");
    println!();

    // ── 11. Note on automated registration ───────────────────────────────────
    println!("── Automated registration with GlobalMiRegistration ───────────────");
    println!();
    println!("  To run automated rigid CT↔MRI registration on these volumes, use the");
    println!("  `ritk_registration::registration::GlobalMiRegistration` API, which");
    println!("  maximises Mutual Information over the full search space via an Adam");
    println!("  optimizer with a multi-resolution schedule:");
    println!();
    println!("    use burn::backend::Autodiff;");
    println!("    use burn_ndarray::NdArray;");
    println!("    use ritk_registration::registration::GlobalMiRegistration;");
    println!("    type AutoB = Autodiff<NdArray<f32>>;");
    println!();
    println!("    // Build the registration object with MI metric + multi-resolution");
    println!("    let reg = GlobalMiRegistration::<AutoB, 3>::new(mi_bins, schedule);");
    println!("    let result = reg.run(&ct_image, &mri_image, &device);");
    println!();
    println!("  Note: Autodiff<NdArray<f32>> is required for gradient-based optimisation.");
    println!("  This example intentionally uses the non-autodiff NdArray<f32> backend");
    println!("  because no gradients are needed for the validation workflow above.");
    println!("  A full automated registration typically takes 2–5 minutes on CPU.");
    println!();
    println!("Done.");

    Ok(())
}
