//! Tests for hit_or_miss
//! Extracted to keep the 500-line structural limit.
use super::*;
use coeus_core::SequentialBackend;
use ritk_image::tensor::Tensor;
use ritk_image::Image as NativeImage;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
type B = coeus_core::SequentialBackend;

#[test]
fn native_transform_matches_legacy_boundary_and_preserves_geometry() {
    let dimensions = [1, 7, 7];
    let mut values = vec![0.0_f32; 49];
    for y in 1..6 {
        for x in 1..6 {
            values[y * 7 + x] = 1.0;
        }
    }
    let origin = Point::new([2.0, 3.0, 5.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::identity();
    let image = NativeImage::from_flat_on(
        values.clone(),
        dimensions,
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .expect("invariant: valid native image");

    let output = HitOrMissTransform::new(1, 0)
        .apply_native(&image, &SequentialBackend)
        .expect("native hit-or-miss succeeds");

    assert_eq!(output.shape(), dimensions);
    assert_eq!(*output.origin(), origin);
    assert_eq!(*output.spacing(), spacing);
    assert_eq!(*output.direction(), direction);
    let legacy = HitOrMissTransform::new(1, 0)
        .apply(&img(values, dimensions))
        .expect("legacy hit-or-miss succeeds");
    assert_eq!(output.data_slice().expect("contiguous output"), vv(&legacy));
    assert_eq!(
        output
            .data_slice()
            .expect("contiguous output")
            .iter()
            .filter(|&&value| value > 0.5)
            .count(),
        9
    );
}

#[test]
fn native_transform_matches_legacy_for_three_dimensional_background_ring() {
    let dimensions = [5, 5, 5];
    let mut values = vec![0.0_f32; 125];
    values[2 * 25 + 2 * 5 + 2] = 1.0;
    let native = NativeImage::from_flat_on(
        values.clone(),
        dimensions,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");

    let native_output = HitOrMissTransform::new(0, 1)
        .apply_native(&native, &SequentialBackend)
        .expect("native hit-or-miss succeeds");
    let legacy_output = HitOrMissTransform::new(0, 1)
        .apply(&img(values, dimensions))
        .expect("legacy hit-or-miss succeeds");

    assert_eq!(
        native_output.data_slice().expect("contiguous output"),
        vv(&legacy_output)
    );
    assert_eq!(
        native_output.data_slice().expect("contiguous output")[2 * 25 + 2 * 5 + 2],
        1.0
    );
}
fn img(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    let t = Tensor::<f32, B>::from_slice(dims, &vals);
    Image::new(
        t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}
fn vv(i: &Image<f32, B, 3>) -> Vec<f32> {
    i.data_slice()
        .expect("invariant: contiguous host storage")
        .to_vec()
}
#[test]
fn test_identity_both_zero() {
    let dims = [6, 6, 6];
    let n = dims[0] * dims[1] * dims[2];
    let v: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
    let r = HitOrMissTransform::new(0, 0)
        .apply(&img(v.clone(), dims))
        .expect("infallible: validated precondition");
    for (i, (&e, &a)) in v.iter().zip(vv(&r).iter()).enumerate() {
        assert!((a - e).abs() < 1e-6, "voxel {i}: expected {e}, got {a}");
    }
}
#[test]
fn test_constant_fg_zeroes() {
    let dims = [8, 8, 8];
    let n = dims[0] * dims[1] * dims[2];
    let out = vv(&HitOrMissTransform::new(0, 1)
        .apply(&img(vec![1.0; n], dims))
        .expect("infallible: validated precondition"));
    for (i, &v) in out.iter().enumerate() {
        assert!(v < 0.5, "voxel {i}={v}");
    }
}
#[test]
fn test_isolated_voxel_detected() {
    let dims = [9, 9, 9];
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut v = vec![0.0_f32; n];
    let c = 4 * ny * nx + 4 * nx + 4;
    v[c] = 1.0;
    let out = vv(&HitOrMissTransform::new(0, 1)
        .apply(&img(v, dims))
        .expect("infallible: validated precondition"));
    assert!(out[c] > 0.5, "centre must be detected, got {}", out[c]);
    for (i, &v) in out.iter().enumerate() {
        if i != c {
            assert!(v < 0.5, "voxel {i}={v}");
        }
    }
}
#[test]
fn test_metadata_preserved() {
    let dims = [5, 5, 5];
    let n = dims[0] * dims[1] * dims[2];
    let t = Tensor::<f32, B>::from_slice(dims, &vec![1.0_f32; n]);
    let o = Point::new([1.0, 2.0, 3.0]);
    let s = Spacing::new([0.5, 0.5, 0.5]);
    let r = HitOrMissTransform::new(0, 0)
        .apply(
            &Image::new(t, o, s, Direction::identity())
                .expect("invariant: fixture tensor has the declared rank"),
        )
        .expect("infallible: validated precondition");
    assert_eq!(*r.origin(), o);
    assert_eq!(*r.spacing(), s);
}
#[test]
fn test_anti_extensivity() {
    let dims = [7, 7, 7];
    let n = dims[0] * dims[1] * dims[2];
    let v: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let out = vv(&HitOrMissTransform::new(1, 1)
        .apply(&img(v.clone(), dims))
        .expect("infallible: validated precondition"));
    for (i, (&orig, &res)) in v.iter().zip(out.iter()).enumerate() {
        assert!(res <= orig + 1e-6, "anti-ext at {i}: res={res}>orig={orig}");
    }
}

/// Regression for the z=1 degenerate-axis trap: a 2-D (z=1) image is a genuine
/// 2-D problem, so the structuring element is 2-D. A 5×5 foreground block (in a
/// 7×7 frame) with fg_radius=1 erodes to its 3×3 interior (9 voxels). Before the
/// fix, every box query failed on the OOB z=±1 neighbours → all-zero output.
#[test]
fn hit_or_miss_works_on_2d_z1_image() {
    let dims = [1, 7, 7];
    let mut v = vec![0.0f32; 7 * 7];
    for y in 1..6 {
        for x in 1..6 {
            v[y * 7 + x] = 1.0;
        }
    }
    let out = vv(&HitOrMissTransform::new(1, 0)
        .apply(&img(v, dims))
        .expect("infallible: validated precondition"));
    let hits = out.iter().filter(|&&x| x > 0.5).count();
    assert_eq!(
        hits, 9,
        "2-D hit-or-miss must detect the 3×3 eroded interior"
    );
}
