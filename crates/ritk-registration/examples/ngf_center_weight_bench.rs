//! Benchmark: uniform vs brain-centroid-weighted NGF rigid registration on a
//! real CT/MR-T1 pair, against a SimpleElastix gold pose.
//!
//! Usage: `cargo run --release -p ritk-registration --example ngf_center_weight_bench -- <ct.nii.gz> <mr.nii.gz>`
//! (defaults to the RIRE-109 pair staged under leoneuro/data).
//!
//! Reports composite rotation, best NGF, generations, and wall time for the
//! uniform metric and the center-weighted metric (`sigma_frac = 0.7`), so the
//! deep-structure (ventricle) alignment improvement is measurable against the
//! elastix reference (composite rotation ≈ 6.50°, 4.6 s on this pair).

use coeus_core::SequentialBackend;
use ritk_filter::BinShrinkImageFilter;
use ritk_io::read_nifti;
use ritk_registration::{
    ct_brain_mask, default_ngf_pyramid, register_rigid_ngf_multires,
    translation_from_centers_of_mass, CtBrainMaskConfig,
};
use std::time::Instant;

type B = SequentialBackend;

/// Composite rotation angle [deg] from a row-major 4×4 rigid matrix.
fn composite_deg(m: &[f64; 16]) -> f64 {
    let tr = m[0] + m[5] + m[10];
    (((tr - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let ct_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "D:/kwavers/leoneuro/data/brain_ct.nii.gz".to_owned());
    let mr_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "D:/kwavers/leoneuro/data/brain_mri_t1.nii.gz".to_owned());

    let device = Default::default();
    let ct = read_nifti::<B, _>(&ct_path, &device)?;
    let mri = read_nifti::<B, _>(&mr_path, &device)?;
    println!(
        "CT  shape {:?} spacing {:?}",
        ct.shape(),
        ct.spacing().to_array()
    );
    println!(
        "MR  shape {:?} spacing {:?}",
        mri.shape(),
        mri.spacing().to_array()
    );

    // Brain mask on the CT (fixed image) restricts NGF to shared rigid anatomy.
    let mask = ct_brain_mask(&ct, &CtBrainMaskConfig::default());

    // Center-of-mass pre-alignment translation (ritk [z,y,x] mm), computed on a
    // coarse downsample (world coords are resolution-independent).
    let coarse = BinShrinkImageFilter::new(vec![1, 4, 4]);
    let com_t = translation_from_centers_of_mass(&coarse.apply(&ct), &coarse.apply(&mri));
    println!("center-of-mass translation seed {com_t:?} mm\n");

    // Multi-resolution coarse→fine NGF, with and without center weighting.
    for frac in [None, Some(0.7_f64)] {
        let pyramid = default_ngf_pyramid(frac);
        let t = Instant::now();
        let (_, res) =
            register_rigid_ngf_multires::<B>(&ct, &mri, [0.0; 3], com_t, Some(&mask), &pyramid);
        let label = if frac.is_some() {
            "multires center-weighted"
        } else {
            "multires uniform        "
        };
        println!(
            "{label}: rot {:.2}°  ngf {:.4}  {:.1}s  rot_deg {:?}  t_mm {:?}",
            composite_deg(&res.matrix),
            res.best_ngf,
            t.elapsed().as_secs_f64(),
            res.rotation_rad
                .map(|r| (r.to_degrees() * 100.0).round() / 100.0),
            res.translation_mm.map(|t| (t * 10.0).round() / 10.0),
        );
    }
    println!("\nelastix gold (reference): composite rotation 6.50°, 4.6 s");
    Ok(())
}
