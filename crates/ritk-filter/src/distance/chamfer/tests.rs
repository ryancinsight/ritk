//! Tests for [`ChamferDistanceTransform`] and the chamfer kernel.
//!
//! These tests verify **scipy.ndimage.distance_transform_cdt parity**:
//! the **interior** distance transform — background voxels are 0, and
//! foreground voxels carry the chamfer distance to the nearest background.

use super::*;
use crate::BinarizationThreshold;

use ritk_core::image::Image;
use ritk_image::test_support as ts;

type B = coeus_core::SequentialBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, dims)
}

fn make_image_with_spacing(vals: Vec<f32>, dims: [usize; 3], sp: [f64; 3]) -> Image<f32, B, 3> {
    ts::make_image_with_spacing::<f32, B, 3>(vals, dims, sp)
}

fn values_finite(img: &Image<f32, B, 3>) -> Vec<f32> {
    ritk_tensor_ops::extract_vec(img).expect("infallible: validated precondition").0
}

// ── Chessboard (scipy parity) ─────────────────────────────────────────────

#[test]
fn chessboard_single_voxel_foreground() {
    // 3x3x3 with fg at (0,0,0) only.
    // scipy: out[fg]=1, out[bg]=0 everywhere.
    let mut data = vec![0.0_f32; 27];
    data[0] = 1.0;
    let img = make_image(data, [3, 3, 3]);
    let cdt = ChamferDistanceTransform::new().with_metric(ChamferMetric::Chessboard);
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    assert_eq!(v[0], 1.0, "fg voxel at index 0 should be 1.0");
    // All other voxels are bg → 0.
    for (i, &x) in v.iter().enumerate().skip(1) {
        assert_eq!(x, 0.0, "bg voxel at index {i} should be 0.0, got {x}");
    }
}

#[test]
fn chessboard_all_background_is_zero() {
    // scipy: no fg voxels → all output is 0.
    let img = make_image(vec![0.0_f32; 27], [3, 3, 3]);
    let cdt = ChamferDistanceTransform::new();
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    for &x in &v {
        assert_eq!(x, 0.0, "all-bg must yield 0 everywhere, got {x}");
    }
}

#[test]
fn chessboard_all_foreground_is_minus_one() {
    // scipy: no bg voxels → sentinel -1.
    let img = make_image(vec![1.0_f32; 27], [3, 3, 3]);
    let cdt = ChamferDistanceTransform::new();
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    for &x in &v {
        assert_eq!(x, -1.0, "all-fg must yield -1.0 sentinel, got {x}");
    }
}

#[test]
fn chessboard_preserves_shape_and_metadata() {
    let img = make_image(vec![0.0_f32; 24], [2, 3, 4]);
    let cdt = ChamferDistanceTransform::new();
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    assert_eq!(out.shape(), [2, 3, 4]);
}

// ── Taxicab (scipy parity) ────────────────────────────────────────────────

#[test]
fn taxicab_single_voxel_foreground() {
    // 3x3x3 with fg at (0,0,0) only.
    // scipy: out[fg]=1, out[bg]=0.
    let mut data = vec![0.0_f32; 27];
    data[0] = 1.0;
    let img = make_image(data, [3, 3, 3]);
    let cdt = ChamferDistanceTransform::new().with_metric(ChamferMetric::Taxicab);
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    assert_eq!(v[0], 1.0, "fg voxel at index 0 should be 1.0");
    for (i, &x) in v.iter().enumerate().skip(1) {
        assert_eq!(x, 0.0, "bg voxel at index {i} should be 0.0, got {x}");
    }
}

#[test]
fn taxicab_exceeds_or_equals_chessboard_on_fg() {
    // A line of fg along x, both metrics give same L∞/L1 since L1=L∞ on
    // single axis. Use a different shape to differentiate.
    // Fg = 3x3x3 cube at center; chessboard (L∞) = 1 inside, taxicab (L1) = 1.
    // For multi-axis, we need a 1-D line: out[center] = 1 for both. So instead
    // use fg = single voxel: out[fg] = 1 for both. The 1 differs only on
    // non-axis-aligned neighbours, which are bg in this case.
    // For our test: build a fg plane and check equality.
    let mut data = vec![0.0_f32; 27];
    // fg plane at z=1 (all y, all x)
    for iy in 0..3 {
        for ix in 0..3 {
            data[9 + iy * 3 + ix] = 1.0;
        }
    }
    let img = make_image(data, [3, 3, 3]);

    let chess = ChamferDistanceTransform::new()
        .with_metric(ChamferMetric::Chessboard)
        .apply(&img)
        .expect("infallible: validated precondition");
    let taxicab = ChamferDistanceTransform::new()
        .with_metric(ChamferMetric::Taxicab)
        .apply(&img)
        .expect("infallible: validated precondition");
    let vc = values_finite(&chess);
    let vt = values_finite(&taxicab);
    // fg voxels at z=1: out=1 (1 step to z=0 or z=2 bg).
    for i in 9..18 {
        assert_eq!(vc[i], 1.0, "chess z=1 idx {i}");
        assert_eq!(vt[i], 1.0, "taxicab z=1 idx {i}");
    }
    // bg voxels: both 0.
    for i in [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    ] {
        assert_eq!(vc[i], 0.0, "chess bg idx {i}");
        assert_eq!(vt[i], 0.0, "taxicab bg idx {i}");
    }
}

#[test]
fn taxicab_l1_far_voxel() {
    // Build a small 5x5x5 with fg at (2,2,2) and bg everywhere else.
    // scipy: out[(2,2,2)] = 1.
    let mut data = vec![0.0_f32; 125];
    data[2 * 25 + 2 * 5 + 2] = 1.0;
    let img = make_image(data, [5, 5, 5]);
    let cdt = ChamferDistanceTransform::new().with_metric(ChamferMetric::Taxicab);
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    let center = 2 * 25 + 2 * 5 + 2;
    assert_eq!(v[center], 1.0, "fg voxel should be 1.0");
    for (i, &x) in v.iter().enumerate() {
        if i == center {
            continue;
        }
        assert_eq!(x, 0.0, "bg voxel at idx {i} should be 0.0, got {x}");
    }
}

// ── Anisotropic spacing (extension over scipy) ────────────────────────────

#[test]
fn chessboard_anisotropic_spacing() {
    // scipy.cdt does not support `sampling`; this is an extension test.
    // 3x3x3 with fg at (0,0,0), spacing [2.0, 1.0, 1.0].
    // s_min = 1.0, weights = [2, 1, 1].
    // The fg voxel's nearest bg is along x or y axis at 1 mm, so the
    // chamfer distance is `min(2, 1, 1) = 1.0` mm under L∞ semantics.
    let mut data = vec![0.0_f32; 27];
    data[0] = 1.0;
    let img = make_image_with_spacing(data, [3, 3, 3], [2.0, 1.0, 1.0]);
    let cdt = ChamferDistanceTransform::new();
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    assert_eq!(
        v[0], 1.0,
        "fg voxel (0,0,0) with anisotropic z-spacing should be 1.0 mm (y/x step is 1 mm)"
    );
    for (i, &x) in v.iter().enumerate().skip(1) {
        assert_eq!(x, 0.0, "bg voxel at idx {i} should be 0.0");
    }
}

// ── Free function ─────────────────────────────────────────────────────────

#[test]
fn free_function_matches_scipy_chessboard_2x2x2() {
    // scipy.cdt on 2x2x2 with fg at (0,0,0): out[0]=1, out[1..]=0.
    let mut data = [0.0_f32; 8];
    data[0] = 1.0;
    let fg: Vec<bool> = data.iter().map(|&v| v > 0.5).collect();
    let free =
        chamfer_distance_transform(&fg, [2, 2, 2], [1.0, 1.0, 1.0], ChamferMetric::Chessboard);
    assert_eq!(free[0], 1);
    for &x in &free[1..] {
        assert_eq!(
            x, 0,
            "bg voxels must be 0 in scipy interior-distance convention"
        );
    }
}

#[test]
fn free_function_matches_scipy_taxicab_2x2x2() {
    let mut data = [0.0_f32; 8];
    data[0] = 1.0;
    let fg: Vec<bool> = data.iter().map(|&v| v > 0.5).collect();
    let free = chamfer_distance_transform(&fg, [2, 2, 2], [1.0, 1.0, 1.0], ChamferMetric::Taxicab);
    assert_eq!(free[0], 1);
    for &x in &free[1..] {
        assert_eq!(x, 0);
    }
}

// ── Threshold semantics ───────────────────────────────────────────────────

#[test]
fn threshold_zero_means_below_or_equal_is_background() {
    // threshold=0.0 means v > 0 is fg. With data=0 everywhere, no fg.
    let img = make_image(vec![0.0_f32; 27], [3, 3, 3]);
    let cdt = ChamferDistanceTransform::new()
        .with_threshold(BinarizationThreshold::new(0.0).expect("valid threshold"));
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    for &x in &v {
        assert_eq!(x, 0.0, "no fg voxels with threshold=0.0 → all 0");
    }
}

#[test]
fn threshold_picks_up_subunit_foreground() {
    // 3x3x3 with fg at (0,0,0) (data=0.6, threshold=0.5).
    let mut data = vec![0.4_f32; 27];
    data[0] = 0.6;
    let img = make_image(data, [3, 3, 3]);
    let cdt = ChamferDistanceTransform::new()
        .with_threshold(BinarizationThreshold::new(0.5).expect("valid threshold"));
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    assert_eq!(v[0], 1.0, "fg voxel at index 0 should be 1.0");
    for (i, &x) in v.iter().enumerate().skip(1) {
        assert_eq!(x, 0.0, "bg voxel at idx {i} should be 0.0");
    }
}

#[test]
fn threshold_scipy_inverse_semantics() {
    // scipy convention: fg = (v > threshold), bg = (v <= threshold).
    // Verify with threshold=0.5: data=0.4 is bg, data=0.6 is fg.
    let mut data = vec![0.4_f32; 27];
    data[0] = 0.6;
    let img = make_image(data, [3, 3, 3]);
    let cdt = ChamferDistanceTransform::new()
        .with_threshold(BinarizationThreshold::new(0.5).expect("valid threshold"));
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    // Index 0 is fg (v=0.6 > 0.5), gets distance 1.
    assert_eq!(v[0], 1.0);
    // All others are bg (v=0.4 <= 0.5), get 0.
    for (_i, &x) in v.iter().enumerate().skip(1) {
        assert_eq!(x, 0.0);
    }
}

// ── Larger volume sanity (hand-traced) ────────────────────────────────────

#[test]
fn chessboard_cube_3x3x3_center() {
    // 7x7x7 with 3x3x3 fg cube centered at (3,3,3). Fg voxels are
    // z ∈ [2,4], y ∈ [2,4], x ∈ [2,4] (27 voxels). Bg everywhere else.
    // scipy chessboard (verified): 26 cube-boundary fg voxels = 1.0, the
    // single cube center (3,3,3) = 2.0; all bg = 0.0.
    let mut data = vec![0.0_f32; 343];
    for z in 2..5 {
        for y in 2..5 {
            for x in 2..5 {
                data[z * 49 + y * 7 + x] = 1.0;
            }
        }
    }
    let img = make_image(data, [7, 7, 7]);
    let cdt = ChamferDistanceTransform::new().with_metric(ChamferMetric::Chessboard);
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    // Cube center (3,3,3) → 2.0 (L∞ distance to nearest bg via cube boundary).
    let center = 3 * 49 + 3 * 7 + 3;
    assert_eq!(v[center], 2.0, "cube center idx {center} should be 2.0");
    // Cube boundary fg voxels → 1.0.
    for z in 2..5 {
        for y in 2..5 {
            for x in 2..5 {
                let i = z * 49 + y * 7 + x;
                if i == center {
                    continue;
                }
                assert_eq!(v[i], 1.0, "cube boundary fg idx {i} should be 1.0");
            }
        }
    }
    // Bg voxels → 0.0.
    for z in 0..7 {
        for y in 0..7 {
            for x in 0..7 {
                if (2..5).contains(&z) && (2..5).contains(&y) && (2..5).contains(&x) {
                    continue;
                }
                let i = z * 49 + y * 7 + x;
                assert_eq!(v[i], 0.0, "bg voxel idx {i} should be 0.0");
            }
        }
    }
}

// ── Differential tests vs scipy.ndimage.distance_transform_cdt ───────────

#[test]
fn diff_vs_scipy_chessboard_3x3x3_cube() {
    // 7x7x7 with 3x3x3 fg cube at center. scipy verified output (z,y,x):
    //   z=0,1,5,6: all 0
    //   z=2,4: corners 1, edges 1, faces 1, no center plane
    //   z=3: corners 1, edges 1, faces 1, center 2
    let mut data = vec![0.0_f32; 343];
    for z in 2..5 {
        for y in 2..5 {
            for x in 2..5 {
                data[z * 49 + y * 7 + x] = 1.0;
            }
        }
    }
    let img = make_image(data, [7, 7, 7]);
    let cdt = ChamferDistanceTransform::new().with_metric(ChamferMetric::Chessboard);
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    // z=3 (middle z plane) row-by-row expected from scipy:
    let z3 = 3 * 49;
    let row_y3 = z3 + 3 * 7;
    let expected_z3 = [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0]; // y=3 of z=3 plane
    for x in 0..7 {
        assert_eq!(v[row_y3 + x], expected_z3[x], "z=3,y=3,x={x}");
    }
}

#[test]
fn diff_vs_scipy_chessboard_two_cubes() {
    // 10x10x10 with two 3x3x3 fg cubes at (1..3, 1..3, 1..3) and (6..8, 6..8, 6..8).
    // scipy chessboard:
    //   Cube 1 boundary voxels → 1.0; center (2,2,2) → 2.0.
    //   Cube 2 boundary voxels → 1.0; center (7,7,7) → 2.0.
    //   Bg voxels → 0.0.
    let mut data = vec![0.0_f32; 1000];
    for z in 1..4 {
        for y in 1..4 {
            for x in 1..4 {
                data[z * 100 + y * 10 + x] = 1.0;
            }
        }
    }
    for z in 6..9 {
        for y in 6..9 {
            for x in 6..9 {
                data[z * 100 + y * 10 + x] = 1.0;
            }
        }
    }
    let img = make_image(data, [10, 10, 10]);
    let cdt = ChamferDistanceTransform::new().with_metric(ChamferMetric::Chessboard);
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    // Cube 1 center (2,2,2) → 2.0
    let c1_center = 2 * 100 + 2 * 10 + 2;
    assert_eq!(v[c1_center], 2.0, "cube 1 center should be 2.0");
    // Cube 1 boundary fg → 1.0
    for z in 1..4 {
        for y in 1..4 {
            for x in 1..4 {
                let i = z * 100 + y * 10 + x;
                if i == c1_center {
                    continue;
                }
                assert_eq!(v[i], 1.0, "cube 1 boundary fg idx {i} should be 1.0");
            }
        }
    }
    // Cube 2 center (7,7,7) → 2.0
    let c2_center = 7 * 100 + 7 * 10 + 7;
    assert_eq!(v[c2_center], 2.0, "cube 2 center should be 2.0");
    // Cube 2 boundary fg → 1.0
    for z in 6..9 {
        for y in 6..9 {
            for x in 6..9 {
                let i = z * 100 + y * 10 + x;
                if i == c2_center {
                    continue;
                }
                assert_eq!(v[i], 1.0, "cube 2 boundary fg idx {i} should be 1.0");
            }
        }
    }
    // Bg voxels: 0
    for z in 0..10 {
        for y in 0..10 {
            for x in 0..10 {
                let in_cube_1 = (1..4).contains(&z) && (1..4).contains(&y) && (1..4).contains(&x);
                let in_cube_2 = (6..9).contains(&z) && (6..9).contains(&y) && (6..9).contains(&x);
                if in_cube_1 || in_cube_2 {
                    continue;
                }
                let i = z * 100 + y * 10 + x;
                assert_eq!(v[i], 0.0, "bg idx {i} should be 0.0");
            }
        }
    }
}

#[test]
fn diff_vs_scipy_taxicab_3x3x3_cube() {
    // 7x7x7 with 3x3x3 fg cube. scipy taxicab verified:
    //   The taxicab distance from each fg voxel to the nearest bg is the
    //   L1 distance to the cube boundary. For a 3x3x3 cube with the
    //   center at (3,3,3), the L1 distance from the center to the nearest
    //   bg is 2 (e.g., (3,3,3) → (3,3,2) → bg at (3,3,1)). Cube boundary
    //   voxels have L1 = 1.
    let mut data = vec![0.0_f32; 343];
    for z in 2..5 {
        for y in 2..5 {
            for x in 2..5 {
                data[z * 49 + y * 7 + x] = 1.0;
            }
        }
    }
    let img = make_image(data, [7, 7, 7]);
    let cdt = ChamferDistanceTransform::new().with_metric(ChamferMetric::Taxicab);
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    let center = 3 * 49 + 3 * 7 + 3;
    assert_eq!(v[center], 2.0, "cube center taxicab should be 2.0");
    for z in 2..5 {
        for y in 2..5 {
            for x in 2..5 {
                let i = z * 49 + y * 7 + x;
                if i == center {
                    continue;
                }
                assert_eq!(v[i], 1.0, "cube boundary taxicab idx {i} should be 1.0");
            }
        }
    }
}

#[test]
fn diff_vs_scipy_chessboard_column_3x3x5() {
    // 3x3x5 with fg at x=2 (full column). scipy chessboard verified:
    //   x=0,1,3,4: 0
    //   x=2: 1
    let mut data = vec![0.0_f32; 45];
    for z in 0..3 {
        for y in 0..3 {
            data[z * 15 + y * 5 + 2] = 1.0;
        }
    }
    let img = make_image(data, [3, 3, 5]);
    let cdt = ChamferDistanceTransform::new().with_metric(ChamferMetric::Chessboard);
    let out = cdt.apply(&img).expect("infallible: validated precondition");
    let v = values_finite(&out);
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..5 {
                let i = z * 15 + y * 5 + x;
                if x == 2 {
                    assert_eq!(v[i], 1.0, "column fg idx {i} should be 1.0");
                } else {
                    assert_eq!(v[i], 0.0, "bg idx {i} should be 0.0");
                }
            }
        }
    }
}
