use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;

type B = NdArray<f32>;

fn img(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, dims)
}

fn vals(image: &Image<B, 3>) -> Vec<f32> {
    image.data_slice().into_owned()
}

/// 1-D roll by +1 along x: [0,1,2,3,4] → [4,0,1,2,3] (last wraps to front).
#[test]
fn shift_x_by_one_wraps() {
    let f = img(vec![0.0, 1.0, 2.0, 3.0, 4.0], [1, 1, 5]);
    let out = CyclicShiftImageFilter::new([0, 0, 1]).apply(&f);
    assert_eq!(vals(&out), vec![4.0, 0.0, 1.0, 2.0, 3.0]);
}

/// Negative shift is the inverse roll; −1 along x → [1,2,3,4,0].
#[test]
fn negative_shift_rolls_other_way() {
    let f = img(vec![0.0, 1.0, 2.0, 3.0, 4.0], [1, 1, 5]);
    let out = CyclicShiftImageFilter::new([0, 0, -1]).apply(&f);
    assert_eq!(vals(&out), vec![1.0, 2.0, 3.0, 4.0, 0.0]);
}

/// A shift equal to a multiple of the axis length is the identity.
#[test]
fn full_period_shift_is_identity() {
    let f = img((0..5).map(|i| i as f32).collect(), [1, 1, 5]);
    let out = CyclicShiftImageFilter::new([0, 0, 10]).apply(&f);
    assert_eq!(vals(&out), vals(&f));
}

/// Cyclic shift is a permutation: every value is preserved (same multiset).
#[test]
fn preserves_all_values() {
    let v: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let f = img(v.clone(), [2, 3, 4]);
    let out = vals(&CyclicShiftImageFilter::new([1, -1, 2]).apply(&f));
    let mut a = v.clone();
    let mut b = out.clone();
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    b.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert_eq!(a, b, "cyclic shift must preserve all values");
}
