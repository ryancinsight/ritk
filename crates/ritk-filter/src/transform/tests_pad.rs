use super::*;
use ritk_image::tensor::Tensor;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, shape)
}

fn voxels(img: &Image<f32, B, 3>) -> Vec<f32> {
    let (v, _) = extract_vec_infallible(img);
    v
}

// ── ConstantPadImageFilter tests ──────────────────────────────────────────

/// Zero-padding: padded voxels filled with 0.
#[test]
fn constant_pad_zero() {
    // 1×1×2 image [3,7], pad by 1 on each side → 1×1×4 [0,3,7,0].
    let img = make_image(vec![3.0, 7.0], [1, 1, 2]);
    let out = ConstantPadImageFilter::new(Padding::new([0, 0, 1]), Padding::new([0, 0, 1]), 0.0)
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 1, 4]);
    let v = voxels(&out);
    assert_eq!(v, vec![0.0, 3.0, 7.0, 0.0]);
}

/// Custom constant pad value.
#[test]
fn constant_pad_custom_value() {
    let img = make_image(vec![5.0], [1, 1, 1]);
    let out = ConstantPadImageFilter::new(Padding::new([0, 0, 2]), Padding::new([0, 0, 2]), -1.0)
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 1, 5]);
    let v = voxels(&out);
    assert_eq!(v, vec![-1.0, -1.0, 5.0, -1.0, -1.0]);
}

/// Constant pad preserves spacing, updates origin.
#[test]
fn constant_pad_origin_updated() {
    let tensor = Tensor::<f32, B>::from_slice([1, 1, 1], &[1.0f32]);
    // Origin at [0, 0, 10], spacing [1, 1, 2] — pad 1 voxel on lower X.
    let img2 = Image::new(
        tensor,
        Point::new([0.0_f64, 0.0, 10.0]),
        Spacing::new([1.0_f64, 1.0, 2.0]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank");
    let out = ConstantPadImageFilter::new(Padding::new([0, 0, 1]), Padding::new([0, 0, 0]), 0.0)
        .apply(&img2)
        .unwrap();
    // Origin x (axis 2) shifts by -1 * spacing[2] = -1 * 2.0 = -2.0 → new origin[2] = 10 - 2 = 8.
    let ox = out.origin()[2];
    assert!((ox - 8.0).abs() < 1e-10, "origin[2]={ox}");
    // Origin z (axis 0) unchanged (pad_lower[0] = 0).
    assert!((out.origin()[0]).abs() < 1e-10, "origin[0] should be 0");
}

#[test]
fn constant_pad_origin_follows_direction_columns() {
    let tensor = Tensor::<f32, B>::from_slice([1, 1, 1], &[1.0f32]);
    let direction = Direction::from_rows([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]);
    let image = Image::new(
        tensor,
        Point::new([10.0, 20.0, 30.0]),
        Spacing::new([2.0, 3.0, 5.0]),
        direction,
    )
    .expect("invariant: fixture tensor has the declared rank");
    let output = ConstantPadImageFilter::new(Padding::new([0, 0, 1]), Padding::zero(), 0.0)
        .apply(&image)
        .expect("constant padding succeeds");

    assert_eq!(
        [output.origin()[0], output.origin()[1], output.origin()[2]],
        [5.0, 20.0, 30.0]
    );
}

#[test]
fn native_constant_pad_preserves_direction_aware_origin() {
    use coeus_core::SequentialBackend;
    use ritk_image::Image as NativeImage;

    let direction = Direction::from_rows([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]);
    let image = NativeImage::from_flat_on(
        vec![3.0],
        [1, 1, 1],
        Point::new([10.0, 20.0, 30.0]),
        Spacing::new([2.0, 3.0, 5.0]),
        direction,
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = ConstantPadImageFilter::new(Padding::new([0, 0, 1]), Padding::zero(), -1.0)
        .apply_native(&image, &SequentialBackend)
        .expect("native constant padding succeeds");

    assert_eq!(output.shape(), [1, 1, 2]);
    assert_eq!(
        output.data_slice().expect("contiguous output"),
        &[-1.0, 3.0]
    );
    assert_eq!(
        [output.origin()[0], output.origin()[1], output.origin()[2]],
        [5.0, 20.0, 30.0]
    );
}

// ── MirrorPadImageFilter tests ────────────────────────────────────────────

/// Mirror pad (ITK symmetric, boundary repeated): 1×1×3 = [1,2,3], pad 2 each
/// side → [2,1,1,2,3,3,2]. Matches `sitk.MirrorPad` (verified) — the boundary
/// voxel `1`/`3` is repeated at the fold, period 2n.
#[test]
fn mirror_pad_1d() {
    let img = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
    let out = MirrorPadImageFilter::new(Padding::new([0, 0, 2]), Padding::new([0, 0, 2]))
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 1, 7]);
    let v = voxels(&out);
    // index -2 → 1 (val 2), -1 → 0 (val 1), [1,2,3], +3 → 2 (val 3), +4 → 1 (val 2)
    let expected = [2.0f32, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0];
    for (i, (&got, exp)) in v.iter().zip(expected).enumerate() {
        assert!((got - exp).abs() < 1e-5, "v[{i}]={got}, expected {exp}");
    }
}

#[test]
fn native_mirror_pad_matches_symmetric_extension() {
    use coeus_core::SequentialBackend;
    use ritk_image::Image as NativeImage;

    let image = NativeImage::from_flat_on(
        vec![1.0, 2.0, 3.0],
        [1, 1, 3],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = MirrorPadImageFilter::new(Padding::new([0, 0, 2]), Padding::new([0, 0, 2]))
        .apply_native(&image, &SequentialBackend)
        .expect("native mirror padding succeeds");
    assert_eq!(
        output.data_slice().expect("contiguous output"),
        &[2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0]
    );
}

/// Mirror index formula for n=1 always returns 0.
#[test]
fn mirror_index_n1() {
    for i in -5i64..=5 {
        assert_eq!(super::mirror_index(i, 1), 0);
    }
}

// ── WrapPadImageFilter tests ──────────────────────────────────────────────

/// Wrap pad: 1×1×3 = [A,B,C], pad 2 on each side → [B,C,A,B,C,A,B].
#[test]
fn wrap_pad_1d() {
    let img = make_image(vec![10.0, 20.0, 30.0], [1, 1, 3]);
    let out = WrapPadImageFilter::new(Padding::new([0, 0, 2]), Padding::new([0, 0, 2]))
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 1, 7]);
    let v = voxels(&out);
    // index shifts: output i → input wrap(i-2, 3)
    // i=0: wrap(-2,3)=1 → 20
    // i=1: wrap(-1,3)=2 → 30
    // i=2: wrap(0,3)=0 → 10
    // i=3: wrap(1,3)=1 → 20
    // i=4: wrap(2,3)=2 → 30
    // i=5: wrap(3,3)=0 → 10
    // i=6: wrap(4,3)=1 → 20
    assert!((v[0] - 20.0).abs() < 1e-5, "v[0]={}", v[0]);
    assert!((v[2] - 10.0).abs() < 1e-5, "v[2]={}", v[2]);
    assert!((v[5] - 10.0).abs() < 1e-5, "v[5]={}", v[5]);
}

#[test]
fn native_wrap_pad_matches_periodic_extension() {
    use coeus_core::SequentialBackend;
    use ritk_image::Image as NativeImage;

    let image = NativeImage::from_flat_on(
        vec![10.0, 20.0, 30.0],
        [1, 1, 3],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = WrapPadImageFilter::new(Padding::new([0, 0, 2]), Padding::new([0, 0, 2]))
        .apply_native(&image, &SequentialBackend)
        .expect("native wrap padding succeeds");
    assert_eq!(
        output.data_slice().expect("contiguous output"),
        &[20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0]
    );
}

/// Output shape correct for wrap pad.
#[test]
fn wrap_pad_shape() {
    let img = make_image(vec![0.0f32; 24], [2, 3, 4]);
    let out = WrapPadImageFilter::new(Padding::new([1, 2, 3]), Padding::new([1, 2, 3]))
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [4, 7, 10]);
}

// ── ZeroFluxNeumannPadImageFilter tests ───────────────────────────────────────

/// Edge-replicate pad of a 1-D ramp: lower pad repeats the first voxel, upper
/// pad repeats the last. Matches ITK ZeroFluxNeumannPad.
#[test]
fn zero_flux_neumann_pad_replicates_edges() {
    let img = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
    let out = ZeroFluxNeumannPadImageFilter::new(Padding::new([0, 0, 2]), Padding::new([0, 0, 1]))
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 1, 6]);
    // [1,1] + [1,2,3] + [3] = repeat edges
    assert_eq!(voxels(&out), vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0]);
}

/// Zero padding on every side is the identity.
#[test]
fn zero_flux_neumann_pad_zero_is_identity() {
    let img = make_image((0..24).map(|i| i as f32).collect(), [2, 3, 4]);
    let out = ZeroFluxNeumannPadImageFilter::new(Padding::new([0, 0, 0]), Padding::new([0, 0, 0]))
        .apply(&img)
        .unwrap();
    assert_eq!(voxels(&out), voxels(&img));
}
