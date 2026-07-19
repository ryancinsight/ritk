use super::*;
use ritk_core::image::Image;
use ritk_image::tensor::Tensor;
use ritk_image::test_support as ts;
use ritk_spatial::{Direction, Point, Spacing};

type B = coeus_core::SequentialBackend;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, shape)
}

fn vals(img: &Image<f32, B, 3>) -> Vec<f32> {
    img.data_slice()
        .expect("invariant: contiguous host storage")
        .to_vec()
}

/// Auto maximum: [1,2,3] → max=3, out=[2,1,0].
#[test]
fn invert_auto_max() {
    let img = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
    let out = InvertIntensityFilter::new().apply(&img);
    let v = vals(&out);
    assert_eq!(
        v,
        vec![2.0_f32, 1.0, 0.0],
        "[1,2,3] with auto max=3 must invert to [2,1,0]"
    );
}

/// Fixed maximum: [1,4,7] with max=10 → [9,6,3].
#[test]
fn invert_fixed_max() {
    let img = make_image(vec![1.0, 4.0, 7.0], [1, 1, 3]);
    let out = InvertIntensityFilter::with_maximum(10.0).apply(&img);
    let v = vals(&out);
    assert_eq!(
        v,
        vec![9.0_f32, 6.0, 3.0],
        "[1,4,7] inverted with max=10 must yield [9,6,3]"
    );
}

/// Minimum maps to (max - min), maximum maps to 0.
#[test]
fn invert_max_maps_to_zero_min_maps_to_range() {
    let img = make_image(vec![2.0, 5.0], [1, 1, 2]);
    let out = InvertIntensityFilter::new().apply(&img);
    let v = vals(&out);
    // auto max = 5.0; 5 - 5 = 0, 5 - 2 = 3
    assert_eq!(v[0], 3.0_f32, "minimum voxel 2 with max=5 → 5-2=3");
    assert_eq!(v[1], 0.0_f32, "maximum voxel 5 with max=5 → 5-5=0");
}

/// Constant image with auto max → all zero.
#[test]
fn invert_constant_auto_max_all_zero() {
    let img = make_image(vec![4.0, 4.0, 4.0], [1, 1, 3]);
    let out = InvertIntensityFilter::new().apply(&img);
    for &v in vals(&out).iter() {
        assert_eq!(v, 0.0_f32, "constant image with auto max → 0 everywhere");
    }
}

/// Spatial metadata is preserved.
#[test]
fn invert_preserves_metadata() {
    let sp = Spacing::new([1.5, 2.5, 3.5]);
    let t = Tensor::<f32, B>::from_slice([1usize, 1, 2], &[1.0_f32, 3.0]);
    let img = Image::new(t, Point::new([0.0, 0.0, 0.0]), sp, Direction::identity())
        .expect("invariant: fixture tensor has the declared rank");
    let out = InvertIntensityFilter::new().apply(&img);
    assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
}

#[test]
fn native_invert_matches_fixed_maximum_contract() {
    use coeus_core::SequentialBackend;
    use ritk_image::Image as NativeImage;

    let image = NativeImage::from_flat_on(
        vec![1.0, 4.0, 7.0],
        [1, 1, 3],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = InvertIntensityFilter::with_maximum(10.0)
        .apply_native(&image, &SequentialBackend)
        .expect("native inversion succeeds");

    assert_eq!(
        output.data_slice().expect("contiguous output"),
        &[9.0, 6.0, 3.0]
    );
}
