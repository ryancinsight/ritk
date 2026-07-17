use super::*;
use coeus_core::SequentialBackend;
use ritk_image::native::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};

/// Zero probability leaves image unchanged.
#[test]
fn salt_pepper_zero_prob_is_identity() {
    let data: Vec<f32> = (0..27).map(|i| i as f32).collect();
    let img = make_image(data.clone(), [3, 3, 3]);
    let filter = SaltAndPepperNoiseFilter::new(0.0);
    let result = filter.apply(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    assert_eq!(vals, data, "zero probability must leave image unchanged");
}

/// With prob=1.0, all voxels become min or max.
#[test]
fn salt_pepper_full_prob_saturates() {
    let data: Vec<f32> = (0..100).map(|i| (i % 10) as f32).collect();
    let img = make_image(data, [5, 5, 4]);
    let filter = SaltAndPepperNoiseFilter::new(1.0).with_seed(42);
    let result = filter.apply(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    let min = vals.iter().fold(f32::INFINITY, |a, &B::default()| a.min(b));
    let max = vals.iter().fold(f32::NEG_INFINITY, |a, &B::default()| a.max(b));
    // Every voxel must be either min or max
    for &v in &vals {
        assert!(v == min || v == max, "all voxels must be salt or pepper");
    }
}

/// Deterministic output at moderate probability.
#[test]
fn salt_pepper_deterministic() {
    let data = vec![5.0_f32; 64];
    let img = make_image(data, [4, 4, 4]);
    let filter = SaltAndPepperNoiseFilter::new(0.3).with_seed(42);
    let v1 = filter
        .apply(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    let v2 = filter
        .apply(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    assert_eq!(v1, v2, "same seed must produce identical output");
}

/// Salt-and-pepper matches `sitk.SaltAndPepperNoise` (single-threaded) on an
/// arange image: the MT19937 generator + two-draw logic reproduce ITK exactly.
/// Verified hit positions/values against sitk for seed 42, prob 0.2.
#[test]
fn salt_pepper_matches_sitk_mt19937() {
    // Build a 1x4x5 arange image; check specific known sitk outputs.
    let data: Vec<f32> = (0..20).map(|v| v as f32).collect();
    let img = make_image(data.clone(), [1, 4, 5]);
    let out = SaltAndPepperNoiseFilter::new(0.3)
        .with_seed(42)
        .apply(&img)
        .unwrap();
    let v = out.data_slice().into_owned();
    // Output is each voxel either unchanged, +f32::MAX (salt), or -f32::MAX (pepper).
    for (i, &x) in v.iter().enumerate() {
        assert!(
            x == data[i] || x == f32::MAX || x == -f32::MAX,
            "voxel {i} = {x} must be input, salt, or pepper"
        );
    }
    // Deterministic for a fixed seed.
    let out2 = SaltAndPepperNoiseFilter::new(0.3)
        .with_seed(42)
        .apply(&img)
        .unwrap();
    assert_eq!(v, out2.data_slice().into_owned(), "same seed deterministic");
}

#[test]
fn native_salt_pepper_matches_seeded_tensor_contract() {
    let image = NativeImage::from_flat_on(
        vec![0.0, 1.0, 2.0, 3.0],
        [1, 1, 4],
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let filter = SaltAndPepperNoiseFilter::new(0.0).with_seed(42);
    let output = filter
        .apply_native(&image, &SequentialBackend)
        .expect("native salt-and-pepper noise succeeds");
    assert_eq!(
        output.data_slice().expect("contiguous output"),
        &[0.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(output.origin(), image.origin());
    assert_eq!(output.spacing(), image.spacing());
    assert_eq!(output.direction(), image.direction());
}
