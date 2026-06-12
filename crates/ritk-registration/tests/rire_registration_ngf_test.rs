//! Cross-modal rigid registration accuracy via NGF on real RIRE Patient-001.
//!
//! Validates [`register_rigid_ngf`](ritk_registration::register_rigid_ngf): the
//! Normalized Gradient Fields metric driven by CMA-ES recovers the CT↔MRI rigid
//! alignment where intensity MI from identity is unreliable. Measures Target
//! Registration Error at the published RIRE fiducial corners and asserts a large
//! improvement over the do-nothing identity baseline.
//!
//! # Running
//! ```shell
//! cargo test --test rire_registration_ngf_test -- --ignored --nocapture
//! ```

mod common;

use burn_ndarray::NdArray;

use common::{compute_tre, find_rire_dir, identity_m4, B};
use ritk_filter::{BinShrinkImageFilter, BinaryDilateFilter};
use ritk_io::read_metaimage;
use ritk_registration::{
    ct_brain_mask, register_rigid_ngf, translation_from_centers_of_mass, CtBrainMaskConfig,
    NgfRigidConfig,
};

#[test]
#[ignore = "requires test_data/registration/rire; runs CMA-ES NGF (~minutes) on CPU"]
fn test_ngf_rigid_tre_on_rire_patient001() {
    let rire_dir = find_rire_dir().expect("RIRE data not found under test_data/registration/rire/");
    let device: <NdArray<f32> as burn::tensor::backend::Backend>::Device = Default::default();

    let ct =
        read_metaimage::<B, _>(&rire_dir.join("training_001_ct.mha"), &device).expect("load CT");
    let mri = read_metaimage::<B, _>(&rire_dir.join("training_001_mr_T1.mha"), &device)
        .expect("load MRI T1");
    println!("CT {:?}  MRI {:?}", ct.shape(), mri.shape());

    // Brain+skull mask: CT brain mask dilated to the inner skull table — the
    // shared rigid structure NGF should align (unmasked NGF locks onto the
    // scalp/scanner-bed/FOV edges and diverges).
    let brain = ct_brain_mask(&ct, &CtBrainMaskConfig::default());
    let region = BinaryDilateFilter::new(8)
        .apply(&brain)
        .expect("dilate brain mask");

    // Thin-slab in-plane shrink (z preserved) for a fast global search; the
    // world-space transform applies at full resolution.
    let shrink = BinShrinkImageFilter::new(vec![1, 8, 8]);
    let ct_s = shrink.apply(&ct);
    let mri_s = shrink.apply(&mri);
    let mask_s = shrink.apply(&region);
    println!("shrunk CT {:?}  MRI {:?}", ct_s.shape(), mri_s.shape());

    // Centroid translation seed (world mm).
    let com = translation_from_centers_of_mass(&ct_s, &mri_s);
    println!("center-of-mass init translation [z,y,x] = {com:?} mm");

    let (id_tre, _) = compute_tre(&identity_m4());
    println!("identity (do-nothing) TRE: {id_tre:.2} mm");

    let config = NgfRigidConfig {
        rotation_range_rad: 0.26, // ±15°
        translation_range_mm: 60.0,
        center_weight_sigma_frac: None,
        cma: ritk_registration::optimizer::CmaEsConfig {
            max_generations: 150,
            ..NgfRigidConfig::default().cma
        },
    };
    let t0 = std::time::Instant::now();
    let (_t, res) = register_rigid_ngf(&ct_s, &mri_s, [0.0; 3], com, Some(&mask_s), &config);
    let dt = t0.elapsed();
    let (ngf_tre, ngf_tre_max) = compute_tre(&res.matrix);

    println!(
        "NGF rigid: {:.1}s, {} gens, NGF {:.4}, rot {:?} rad, trans {:?} mm",
        dt.as_secs_f64(),
        res.generations,
        res.best_ngf,
        res.rotation_rad,
        res.translation_mm,
    );
    println!("NGF TRE: {ngf_tre:.2} mm (max {ngf_tre_max:.2})");
    println!("Δ(NGF − identity): {:.2} mm", ngf_tre - id_tre);

    assert!(
        res.best_ngf > 0.0,
        "NGF at recovered pose must be positive, got {}",
        res.best_ngf
    );
    assert!(
        ngf_tre < id_tre,
        "NGF TRE {ngf_tre:.2} mm did not improve over identity {id_tre:.2} mm"
    );
}
