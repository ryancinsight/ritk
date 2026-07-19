use super::MorphologicalWatershed;
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::test_support as ts;
use ritk_image::Image as NativeImage;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = SequentialBackend;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}

/// A W-shaped 1-D relief has two minima (x=2, x=6) split by a ridge at x=4.
/// Marker-less watershed seeds a basin at each minimum and floods until they
/// collide on the ridge, which becomes a watershed line (label 0).
#[test]
fn morphological_watershed_two_basins_with_ridge() {
    let vals = vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
    let out = MorphologicalWatershed::new(0.0)
        .unwrap()
        .apply(&make(vals, [1, 1, 9]))
        .unwrap();
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(ov, vec![1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0, 2.0]);
}

/// A flat (constant) relief is a single regional minimum → one basin, no lines.
#[test]
fn morphological_watershed_flat_is_single_basin() {
    let dims = [1usize, 4, 5];
    let n: usize = dims.iter().product();
    let out = MorphologicalWatershed::default()
        .apply(&make(vec![3.0; n], dims))
        .unwrap();
    let (ov, _) = extract_vec_infallible(&out);
    assert!(ov.iter().all(|&v| v == 1.0), "constant relief is one basin");
}

/// A positive `level` merges shallow minima: a tiny dip (depth 1) below the
/// `level` threshold is filled by h-minima, so only the deep basin remains.
#[test]
fn morphological_watershed_level_merges_shallow_minima() {
    // Deep well at x=2 (depth 5) and a shallow dip at x=6 (depth 1).
    let vals = vec![5.0, 2.0, 0.0, 2.0, 5.0, 5.0, 4.0, 5.0, 5.0];
    let out = MorphologicalWatershed::new(2.0)
        .unwrap()
        .apply(&make(vals, [1, 1, 9]))
        .unwrap();
    let (ov, _) = extract_vec_infallible(&out);
    // The depth-1 dip (< level 2) is suppressed; a single basin remains (no
    // watershed line splitting it off).
    assert!(ov.iter().all(|&v| v == 1.0 || v == 0.0));
    assert_eq!(
        ov.iter().filter(|&&v| v == 1.0).count(),
        9,
        "one merged basin, no line"
    );
}

#[test]
fn morphological_watershed_native_matches_legacy_at_all_levels() {
    let values = vec![5.0, 2.0, 0.0, 2.0, 5.0, 5.0, 4.0, 5.0, 5.0];
    let legacy = make(values.clone(), [1, 1, 9]);
    let origin = Point::new([2.0, 3.0, 5.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::identity();
    let native = NativeImage::from_flat_on(
        values,
        [1, 1, 9],
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .unwrap();
    for level in [0.0, 1.0, 2.0] {
        let filter = MorphologicalWatershed::new(level).unwrap();
        assert_eq!(filter.level(), level);
        let expected = filter.apply(&legacy).unwrap();
        let actual = filter.apply_native(&native, &SequentialBackend).unwrap();
        assert_eq!(
            actual.data_slice().unwrap(),
            expected
                .data_slice()
                .expect("invariant: contiguous host storage")
        );
        assert_eq!(*actual.origin(), origin);
        assert_eq!(*actual.spacing(), spacing);
        assert_eq!(*actual.direction(), direction);
    }
}

#[test]
fn morphological_watershed_rejects_invalid_levels() {
    for level in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
        assert_eq!(
            MorphologicalWatershed::new(level).unwrap_err().to_string(),
            format!("morphological watershed level must be finite and nonnegative, got {level}")
        );
    }
}
