//! Golden label maps pinning SLIC's exact output, tie-breaking included.
//!
//! The fixtures mix smooth ramps with constant plateaus so that many voxels
//! see two candidate centers at equal distance; the winner is decided by the
//! order in which the search index lists candidates, so any reordering of that
//! index changes the hash.

use super::*;
use coeus_core::SequentialBackend;
use ritk_image::test_support::make_image;

/// FNV-1a over the label bit patterns: an order- and bit-sensitive digest.
fn label_digest(labels: &[f32]) -> u64 {
    labels
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, label| {
            (hash ^ u64::from(label.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// Plateaus of width `plateau` along the first axis plus a ramp on the last.
fn plateau_field(len: usize, plateau: usize, last_extent: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "fixture indices stay far below f32's exact-integer range"
            )]
            let level = ((i / plateau) % 5) as f32 * 40.0 + (i % last_extent) as f32 * 0.5;
            level
        })
        .collect()
}

fn run<const D: usize>(values: Vec<f32>, dims: [usize; D], superpixels: usize) -> Vec<f32> {
    let image = make_image::<f32, SequentialBackend, D>(values, dims);
    let config = SlicConfig::new(superpixels)
        .and_then(|config| config.with_max_iterations(6))
        .expect("invariant: 6 iterations and a nonzero count are valid");
    let output = SlicSuperpixelFilter::new(config)
        .apply(&image)
        .expect("invariant: fixture is finite, 2-D/3-D, and nonempty");
    output.data().to_vec()
}

#[test]
fn two_dimensional_plateau_labels_match_the_golden_digest() {
    let labels = run(plateau_field(48 * 40, 97, 40), [48, 40], 36);
    assert!(
        labels.iter().any(|label| *label != labels[0]),
        "fixture must split"
    );
    assert_eq!(label_digest(&labels), GOLDEN_2D, "2-D label map changed");
}

#[test]
fn three_dimensional_plateau_labels_match_the_golden_digest() {
    let labels = run(plateau_field(20 * 18 * 16, 311, 16), [20, 18, 16], 48);
    assert!(
        labels.iter().any(|label| *label != labels[0]),
        "fixture must split"
    );
    assert_eq!(label_digest(&labels), GOLDEN_3D, "3-D label map changed");
}

const GOLDEN_2D: u64 = 17_373_636_990_762_219_813;
const GOLDEN_3D: u64 = 2_041_448_085_080_745_253;
