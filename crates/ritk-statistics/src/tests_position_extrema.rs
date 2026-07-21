use super::*;
use coeus_core::MoiraiBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type TestBackend = MoiraiBackend;

fn make_image_1d(data: Vec<f32>) -> Image<f32, TestBackend, 1> {
    let dims = [data.len()];
    make_image(data, dims)
}

fn make_image<const D: usize>(data: Vec<f32>, dims: [usize; D]) -> Image<f32, TestBackend, D> {
    Image::from_flat(
        data,
        dims,
        Point::new([0.0; D]),
        Spacing::new([1.0; D]),
        Direction::identity(),
    )
    .expect("invariant: test image shape matches its voxel count")
}

// ── 1-D ────────────────────────────────────────────────────────────────────

#[test]
fn minimum_position_1d_unique() {
    let img = make_image_1d(vec![3.0, 1.0, 4.0, 1.5, 9.0]);
    assert_eq!(
        minimum_position(&img).expect("infallible: validated precondition"),
        Some([1])
    );
}

#[test]
fn maximum_position_1d_unique() {
    let img = make_image_1d(vec![3.0, 1.0, 4.0, 1.5, 9.0]);
    assert_eq!(
        maximum_position(&img).expect("infallible: validated precondition"),
        Some([4])
    );
}

#[test]
fn minimum_position_1d_tie_breaks_to_lowest_index() {
    let img = make_image_1d(vec![2.0, 1.0, 3.0, 1.0, 4.0]);
    // Two minima at indices 1 and 3; lowest wins.
    assert_eq!(
        minimum_position(&img).expect("infallible: validated precondition"),
        Some([1])
    );
}

#[test]
fn maximum_position_1d_tie_breaks_to_lowest_index() {
    let img = make_image_1d(vec![5.0, 9.0, 1.0, 9.0, 4.0]);
    assert_eq!(
        maximum_position(&img).expect("infallible: validated precondition"),
        Some([1])
    );
}

#[test]
fn minimum_position_1d_at_index_zero() {
    let img = make_image_1d(vec![-5.0, 1.0, 2.0, 3.0]);
    assert_eq!(
        minimum_position(&img).expect("infallible: validated precondition"),
        Some([0])
    );
}

// ── 3-D row-major layout ───────────────────────────────────────────────────

#[test]
fn minimum_position_3d_simple() {
    // 2×2×2 image: min is at (iz=1, iy=0, ix=1) → flat index = 1*4 + 0*2 + 1 = 5
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, -7.0, 7.0, 8.0];
    let img = make_image(data, [2, 2, 2]);
    assert_eq!(
        minimum_position(&img).expect("infallible: validated precondition"),
        Some([1, 0, 1])
    );
}

#[test]
fn maximum_position_3d_simple() {
    // 2×2×2 image: max is 99.0 at flat index 6 = (1, 1, 0)
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, -7.0, 99.0, 8.0];
    let img = make_image(data, [2, 2, 2]);
    assert_eq!(
        maximum_position(&img).expect("infallible: validated precondition"),
        Some([1, 1, 0])
    );
}

#[test]
fn minimum_position_3d_first_voxel() {
    // Min at flat index 0
    let mut data = vec![1.0_f32; 27];
    data[0] = -100.0;
    let img = make_image(data, [3, 3, 3]);
    assert_eq!(
        minimum_position(&img).expect("infallible: validated precondition"),
        Some([0, 0, 0])
    );
}

#[test]
fn minimum_position_3d_last_voxel() {
    // Min at flat index 26 = (2, 2, 2)
    let mut data = vec![1.0_f32; 27];
    data[26] = -100.0;
    let img = make_image(data, [3, 3, 3]);
    assert_eq!(
        minimum_position(&img).expect("infallible: validated precondition"),
        Some([2, 2, 2])
    );
}

#[test]
fn minimum_position_3d_tie_breaks_to_lowest_flat() {
    // Two minima at flat 3 = (0, 1, 0) and flat 12 = (1, 1, 0)
    // Lowest flat wins: 3 → (0, 1, 0)
    let mut data = vec![5.0_f32; 27];
    data[3] = -10.0; // (0, 1, 0)
    data[12] = -10.0; // (1, 1, 0)
    let img = make_image(data, [3, 3, 3]);
    assert_eq!(
        minimum_position(&img).expect("infallible: validated precondition"),
        Some([0, 1, 0])
    );
}

#[test]
fn minimum_position_3d_constant() {
    let img = make_image(vec![7.0_f32; 27], [3, 3, 3]);
    // All tied at value 7 → lowest flat index wins → (0, 0, 0)
    assert_eq!(
        minimum_position(&img).expect("infallible: validated precondition"),
        Some([0, 0, 0])
    );
    assert_eq!(
        maximum_position(&img).expect("infallible: validated precondition"),
        Some([0, 0, 0])
    );
}

#[test]
fn maximum_position_3d_single_voxel() {
    let img = make_image(vec![42.0_f32], [1, 1, 1]);
    assert_eq!(
        minimum_position(&img).expect("infallible: validated precondition"),
        Some([0, 0, 0])
    );
    assert_eq!(
        maximum_position(&img).expect("infallible: validated precondition"),
        Some([0, 0, 0])
    );
}

#[test]
fn minimum_position_3d_negative_values() {
    let data = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
    let img = make_image(data, [2, 2, 2]);
    // Min is -8 at flat 7 = (1, 1, 1)
    assert_eq!(
        minimum_position(&img).expect("infallible: validated precondition"),
        Some([1, 1, 1])
    );
}

#[test]
fn flat_to_multi_round_trip() {
    // Spot-check: every flat index in a 2×3×4 volume maps correctly.
    let dims = [2_usize, 3, 4];
    for flat in 0..(2 * 3 * 4) {
        let m = flat_to_multi(flat, dims);
        let mut recovered = 0_usize;
        for (k, &mk) in m.iter().enumerate() {
            let mut stride = 1;
            for &d in dims.iter().skip(k + 1) {
                stride *= d;
            }
            recovered += mk * stride;
        }
        assert_eq!(recovered, flat, "flat={} → {:?} → {}", flat, m, recovered);
    }
}
