//! Tests for voting_binary
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;

type B = LegacyBurnBackend;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::burn_compat::make_image::<B, 3>(data, shape)
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

/// All-background image with birth_threshold=1 and high survival_threshold=0:
/// no neighbours → fg_count=0 < birth_threshold → stays background.
#[test]
fn all_background_stays_background() {
    let img = make_image(vec![0.0f32; 27], [3, 3, 3]);
    let filter = VotingBinaryImageFilter::default(); // birth=1
    let out = filter.apply(&img).unwrap();
    // No foreground neighbours anywhere → all voxels stay background.
    assert!(voxels(&out).iter().all(|&v| v == 0.0));
}

/// All-foreground image: every voxel has neighbours ≥ survival_threshold=1 → stays foreground.
#[test]
fn all_foreground_survives() {
    let img = make_image(vec![1.0f32; 27], [3, 3, 3]);
    let filter = VotingBinaryImageFilter::default(); // survival=1
    let out = filter.apply(&img).unwrap();
    // Every fg voxel has at least 1 fg neighbour → survives.
    assert!(voxels(&out).iter().all(|&v| (v - 1.0).abs() < 1e-5));
}

/// Single isolated foreground voxel: survives at survival_threshold=1 (counts itself),
/// but dies at survival_threshold=2 (needs at least one neighbor).
#[test]
fn isolated_fg_behavior() {
    let mut data = vec![0.0f32; 27];
    data[13] = 1.0; // center voxel
    let img = make_image(data, [3, 3, 3]);

    // survival_threshold = 1 -> survives
    let filter1 = VotingBinaryImageFilter::new(1, 1, 1, 1.0, 0.0);
    let out1 = filter1.apply(&img).unwrap();
    assert_eq!(voxels(&out1)[13], 1.0);

    // survival_threshold = 2 -> dies
    let filter2 = VotingBinaryImageFilter::new(1, 1, 2, 1.0, 0.0);
    let out2 = filter2.apply(&img).unwrap();
    assert_eq!(voxels(&out2)[13], 0.0);
}

/// Birth: background voxel adjacent to a foreground cluster is born
/// when fg_count >= birth_threshold.
#[test]
fn birth_from_fg_neighbor() {
    // 1×1×3: [1, 1, 0]. With r=1, birth=1:
    // voxel 2 (bg): neighbours = [1,1] → fg_count=2 ≥ 1 → born.
    let img = make_image(vec![1.0, 1.0, 0.0], [1, 1, 3]);
    let filter = VotingBinaryImageFilter::new(1, 1, 1, 1.0, 0.0);
    let out = filter.apply(&img).unwrap();
    let v = voxels(&out);
    assert!(
        (v[2] - 1.0).abs() < 1e-5,
        "voxel 2 should be born, got {}",
        v[2]
    );
}

/// Spatial metadata preserved.
#[test]
fn preserves_metadata() {
    let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
    let out = VotingBinaryImageFilter::default().apply(&img).unwrap();
    assert_eq!(out.shape(), [2, 2, 2]);
    assert_eq!(*out.origin(), *img.origin());
}
