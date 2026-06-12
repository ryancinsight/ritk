use super::compute::flat_to_multi;
use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type TestBackend = NdArray<f32>;

fn make_image_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
    let n = data.len();
    let device = Default::default();
    let tensor =
        Tensor::<TestBackend, 1>::from_data(TensorData::new(data, Shape::new([n])), &device);
    Image::new(
        tensor,
        Point::new([0.0]),
        Spacing::new([1.0]),
        Direction::identity(),
    )
}

fn make_image_2d(data: Vec<f32>, dims: [usize; 2]) -> Image<TestBackend, 2> {
    let device = Default::default();
    let tensor =
        Tensor::<TestBackend, 2>::from_data(TensorData::new(data, Shape::new(dims)), &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0]),
        Spacing::new([1.0, 1.0]),
        Direction::identity(),
    )
}

fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
    let device = Default::default();
    let tensor =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

// ── 1-D ────────────────────────────────────────────────────────────────────

#[test]
fn value_indices_1d_basic() {
    // [10, 20, 10, 30, 20] → 10: [0,2]; 20: [1,4]; 30: [3]
    let img = make_image_1d(vec![10.0, 20.0, 10.0, 30.0, 20.0]);
    let vi = value_indices(&img, None);
    assert_eq!(vi.num_distinct(), 3);
    assert_eq!(vi.total(), 5);
    let exp_10: &[[usize; 1]] = &[[0], [2]];
    let exp_20: &[[usize; 1]] = &[[1], [4]];
    let exp_30: &[[usize; 1]] = &[[3]];
    assert_eq!(vi.get(10.0), Some(exp_10));
    assert_eq!(vi.get(20.0), Some(exp_20));
    assert_eq!(vi.get(30.0), Some(exp_30));
    assert_eq!(vi.get(99.0), None);
}

#[test]
fn value_indices_1d_constant() {
    let img = make_image_1d(vec![7.0; 4]);
    let vi = value_indices(&img, None);
    assert_eq!(vi.num_distinct(), 1);
    assert_eq!(vi.total(), 4);
    let exp: &[[usize; 1]] = &[[0], [1], [2], [3]];
    assert_eq!(vi.get(7.0), Some(exp));
}

#[test]
fn value_indices_1d_single_voxel() {
    let img = make_image_1d(vec![42.0]);
    let vi = value_indices(&img, None);
    assert_eq!(vi.num_distinct(), 1);
    let exp: &[[usize; 1]] = &[[0]];
    assert_eq!(vi.get(42.0), Some(exp));
}

#[test]
fn value_indices_1d_ignore_value() {
    let img = make_image_1d(vec![1.0, 2.0, 1.0, 3.0, 1.0]);
    let vi = value_indices(&img, Some(1.0));
    assert_eq!(vi.num_distinct(), 2);
    assert_eq!(vi.total(), 2);
    assert_eq!(vi.get(1.0), None);
    let exp_2: &[[usize; 1]] = &[[1]];
    let exp_3: &[[usize; 1]] = &[[3]];
    assert_eq!(vi.get(2.0), Some(exp_2));
    assert_eq!(vi.get(3.0), Some(exp_3));
}

// ── 2-D (scipy docstring example) ─────────────────────────────────────────

#[test]
fn value_indices_2d_docstring_example() {
    // 6×6 array from the scipy.ndimage.value_indices docstring.
    // [[2 2 2 0 0 3] 3
    // [2 2 2 0 0 0]
    // [0 0 1 1 0 0]
    // [0 0 1 1 0 0]
    // [0 0 0 0 1 0]
    // [0 0 0 0 0 0]]
    let mut a = vec![0.0_f32; 36];
    // value 2 block: [0:2, 0:3] and [1, 0:3]
    for r in 0..2 {
        for c in 0..3 {
            a[r * 6 + c] = 2.0;
        }
    }
    // value 3: (0, 5)
    a[5] = 3.0;
    // value 1 block: [2:4, 2:4]
    for r in 2..4 {
        for c in 2..4 {
            a[r * 6 + c] = 1.0;
        }
    }
    // value 1: (4, 4)
    a[4 * 6 + 4] = 1.0;
    let img = make_image_2d(a, [6, 6]);
    let vi = value_indices(&img, None);

    assert_eq!(vi.num_distinct(), 4);
    assert_eq!(vi.total(), 36);

    // 1.0 at (2,2), (2,3), (3,2), (3,3), (4,4)
    let exp_1: &[[usize; 2]] = &[[2, 2], [2, 3], [3, 2], [3, 3], [4, 4]];
    assert_eq!(vi.get(1.0), Some(exp_1));
    // 2.0 at (0,0..3), (1,0..3) — row-major
    let exp_2: &[[usize; 2]] = &[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]];
    assert_eq!(vi.get(2.0), Some(exp_2));
    // 3.0 at (0, 5)
    let exp_3: &[[usize; 2]] = &[[0, 5]];
    assert_eq!(vi.get(3.0), Some(exp_3));
    // 0.0 fills the rest — verify count = 36 - 6 - 5 - 1 = 24
    assert_eq!(vi.len(0.0), 24);
}

#[test]
fn value_indices_2d_ignore_value() {
    let mut a = vec![0.0_f32; 36];
    a[5] = 3.0;
    for r in 2..4 {
        for c in 2..4 {
            a[r * 6 + c] = 1.0;
        }
    }
    let img = make_image_2d(a, [6, 6]);
    let vi = value_indices(&img, Some(0.0));
    assert_eq!(vi.num_distinct(), 2);
    assert_eq!(vi.total(), 5);
    assert_eq!(vi.get(0.0), None);
    let exp_3: &[[usize; 2]] = &[[0, 5]];
    assert_eq!(vi.get(3.0), Some(exp_3));
}

// ── 3-D ────────────────────────────────────────────────────────────────────

#[test]
fn value_indices_3d_two_corner_voxels_and_center() {
    // 3×3×3 with 1.0 at (0,0,0) and (2,2,2), 5.0 at (1,1,1), rest = 0.0.
    let mut a = vec![0.0_f32; 27];
    a[0] = 1.0; // (0,0,0)
    a[26] = 1.0; // (2,2,2)
    a[13] = 5.0; // (1,1,1)
    let img = make_image_3d(a, [3, 3, 3]);
    let vi = value_indices(&img, None);

    assert_eq!(vi.num_distinct(), 3);
    assert_eq!(vi.total(), 27);

    let exp_1: &[[usize; 3]] = &[[0, 0, 0], [2, 2, 2]];
    let exp_5: &[[usize; 3]] = &[[1, 1, 1]];
    assert_eq!(vi.get(1.0), Some(exp_1));
    assert_eq!(vi.get(5.0), Some(exp_5));
    assert_eq!(vi.len(0.0), 24);
}

#[test]
fn value_indices_3d_all_same_value() {
    // 2×2×2 filled with 7.0 → 8 occurrences of a single value.
    let img = make_image_3d(vec![7.0_f32; 8], [2, 2, 2]);
    let vi = value_indices(&img, None);
    assert_eq!(vi.num_distinct(), 1);
    assert_eq!(vi.total(), 8);
    let exp: &[[usize; 3]] = &[
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ];
    assert_eq!(vi.get(7.0), Some(exp));
}

#[test]
fn value_indices_3d_single_voxel() {
    let img = make_image_3d(vec![42.0_f32], [1, 1, 1]);
    let vi = value_indices(&img, None);
    assert_eq!(vi.num_distinct(), 1);
    let exp: &[[usize; 3]] = &[[0, 0, 0]];
    assert_eq!(vi.get(42.0), Some(exp));
}

#[test]
fn value_indices_3d_ignore_value_excludes_voxels() {
    // 2×3×4 with 6 distinct non-zero values placed at known locations;
    // ignore_value=0.0 drops the 18 zero voxels from the output.
    let mut a = vec![0.0_f32; 24];
    a[0] = 1.0; // (0,0,0)
    a[1] = 2.0; // (0,0,1)
    a[4] = 3.0; // (0,1,0)
    a[5] = 4.0; // (0,1,1)
    a[12] = 5.0; // (1,0,0)
    a[23] = 6.0; // (1,2,3)
    let img = make_image_3d(a, [2, 3, 4]);

    let vi_full = value_indices(&img, None);
    assert_eq!(vi_full.num_distinct(), 7);
    assert_eq!(vi_full.total(), 24);

    let vi_ignore = value_indices(&img, Some(0.0));
    assert_eq!(vi_ignore.num_distinct(), 6);
    assert_eq!(vi_ignore.total(), 6);
    assert_eq!(vi_ignore.get(0.0), None);
    let exp_1: &[[usize; 3]] = &[[0, 0, 0]];
    let exp_2: &[[usize; 3]] = &[[0, 0, 1]];
    let exp_3: &[[usize; 3]] = &[[0, 1, 0]];
    let exp_4: &[[usize; 3]] = &[[0, 1, 1]];
    let exp_5: &[[usize; 3]] = &[[1, 0, 0]];
    let exp_6: &[[usize; 3]] = &[[1, 2, 3]];
    assert_eq!(vi_ignore.get(1.0), Some(exp_1));
    assert_eq!(vi_ignore.get(2.0), Some(exp_2));
    assert_eq!(vi_ignore.get(3.0), Some(exp_3));
    assert_eq!(vi_ignore.get(4.0), Some(exp_4));
    assert_eq!(vi_ignore.get(5.0), Some(exp_5));
    assert_eq!(vi_ignore.get(6.0), Some(exp_6));
}

#[test]
fn value_indices_3d_ignore_value_not_present() {
    // 2×2×2, all zero except one 1.0; ignore_value=999 has no effect.
    let mut a = vec![0.0_f32; 8];
    a[0] = 1.0;
    let img = make_image_3d(a, [2, 2, 2]);
    let vi = value_indices(&img, Some(999.0));
    assert_eq!(vi.num_distinct(), 2);
    assert_eq!(vi.total(), 8);
    let zeros: &[[usize; 3]] = &[
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ];
    assert_eq!(vi.get(0.0), Some(zeros));
    let exp_1: &[[usize; 3]] = &[[0, 0, 0]];
    assert_eq!(vi.get(1.0), Some(exp_1));
}

// ── Properties / invariants ────────────────────────────────────────────────

#[test]
fn value_indices_3d_row_major_ordering() {
    // 2×2×2 with values 1..=8 in flat order; verify that
    // value_indices returns each value at the corresponding
    // multi-index without reordering.
    let img = make_image_3d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2]);
    let vi = value_indices(&img, None);
    assert_eq!(vi.num_distinct(), 8);
    assert_eq!(vi.total(), 8);
    for v in 1..=8u32 {
        let value = v as f32;
        let expected = flat_to_multi((v - 1) as usize, [2, 2, 2]);
        let exp: &[[usize; 3]] = &[expected];
        assert_eq!(vi.get(value), Some(exp));
    }
}

#[test]
fn value_indices_3d_total_equals_voxel_count_without_ignore() {
    // Random-ish pattern; verify total() equals n.
    let mut a = Vec::with_capacity(125);
    for i in 0..125 {
        a.push(((i * 7 + 3) % 5) as f32);
    }
    let img = make_image_3d(a, [5, 5, 5]);
    let vi = value_indices(&img, None);
    assert_eq!(vi.total(), 125);
}

#[test]
fn value_indices_3d_total_equals_n_minus_ignored_count() {
    // 3×3×3 with 9 voxels of value 5.0; ignore them and verify total
    // drops by 9.
    let mut a = vec![0.0_f32; 27];
    for flat in (0..27).step_by(3) {
        a[flat] = 5.0; // 9 voxels: 0, 3, 6, ..., 24
    }
    let img = make_image_3d(a, [3, 3, 3]);
    let vi_full = value_indices(&img, None);
    assert_eq!(vi_full.total(), 27);
    assert_eq!(vi_full.len(5.0), 9);
    let vi_ignored = value_indices(&img, Some(5.0));
    assert_eq!(vi_ignored.total(), 18);
    assert_eq!(vi_ignored.get(5.0), None);
}

#[test]
fn flat_to_multi_round_trip_3d() {
    // Spot-check: every flat index in a 2×3×4 volume maps to a
    // multi-index that recovers the original flat index.
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

#[test]
fn f32_key_bit_equality() {
    // 0.0 and -0.0 are distinct bit patterns → distinct keys.
    let k_pos = F32Key::new(0.0_f32);
    let k_neg = F32Key::new(-0.0_f32);
    assert_ne!(k_pos, k_neg);
    assert_eq!(k_pos, F32Key::new(0.0_f32));
    // Bit-equal copies are equal
    assert_eq!(k_pos, F32Key::new(0.0_f32));
}
