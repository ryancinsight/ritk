use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_core::image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type TestBackend = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
    let device = Default::default();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn values(img: &Image<TestBackend, 3>) -> Vec<f32> {
    ritk_tensor_ops::extract_vec(img).unwrap().0
}

#[test]
fn prewitt_constant_image_has_zero_gradient() {
    let img = make_image(vec![5.0_f32; 27], [3, 3, 3]);
    let filt = PrewittFilter::unit();
    let out = filt.apply(&img).unwrap();
    let v = values(&out);
    assert!(
        v.iter().all(|&x| x.abs() < 1e-5),
        "constant image should produce zero gradient, got max |v| = {}",
        v.iter().fold(0.0_f32, |a, &b| a.max(b.abs()))
    );
}

#[test]
fn prewitt_x_ramp_recovers_unit_gradient() {
    // I(x) = x with unit spacing
    let mut vals = Vec::with_capacity(27);
    for _iz in 0..3 {
        for _iy in 0..3 {
            for ix in 0..3 {
                vals.push(ix as f32);
            }
        }
    }
    let img = make_image(vals, [3, 3, 3]);
    let filt = PrewittFilter::unit();
    let (_gz, _gy, gx) = filt.apply_components(&img).unwrap();
    let gx_v = values(&gx);
    // Interior voxel (iz=1, iy=1, ix=1): dI/dx = 1
    let interior_idx = 9 + 3 + 1;
    assert!(
        (gx_v[interior_idx] - 1.0).abs() < 1e-5,
        "interior x-gradient should be 1.0, got {}",
        gx_v[interior_idx]
    );
    // Boundary voxel at ix=0 with replicate padding yields a one-sided
    // difference: raw = I[1] − I[0] = 1, smoothed twice by [1,1,1] gives
    // 9, normalized to 9/18 = 0.5.
    let boundary_idx = 9 + 3;
    assert!(
        (gx_v[boundary_idx] - 0.5).abs() < 1e-5,
        "boundary x-gradient (one-sided) should be 0.5, got {}",
        gx_v[boundary_idx]
    );
}

#[test]
fn prewitt_y_ramp_recovers_unit_gradient() {
    // I(y) = y with unit spacing
    let mut vals = Vec::with_capacity(27);
    for _iz in 0..3 {
        for iy in 0..3 {
            for _ix in 0..3 {
                vals.push(iy as f32);
            }
        }
    }
    let img = make_image(vals, [3, 3, 3]);
    let filt = PrewittFilter::unit();
    let (_gz, gy, _gx) = filt.apply_components(&img).unwrap();
    let gy_v = values(&gy);
    let interior_idx = 9 + 3 + 1;
    assert!(
        (gy_v[interior_idx] - 1.0).abs() < 1e-5,
        "interior y-gradient should be 1.0, got {}",
        gy_v[interior_idx]
    );
}

#[test]
fn prewitt_z_ramp_recovers_unit_gradient() {
    // I(z) = z with unit spacing
    let mut vals = Vec::with_capacity(27);
    for iz in 0..3 {
        for _iy in 0..3 {
            for _ix in 0..3 {
                vals.push(iz as f32);
            }
        }
    }
    let img = make_image(vals, [3, 3, 3]);
    let filt = PrewittFilter::unit();
    let (gz, _gy, _gx) = filt.apply_components(&img).unwrap();
    let gz_v = values(&gz);
    let interior_idx = 9 + 3 + 1;
    assert!(
        (gz_v[interior_idx] - 1.0).abs() < 1e-5,
        "interior z-gradient should be 1.0, got {}",
        gz_v[interior_idx]
    );
}

#[test]
fn prewitt_magnitude_zero_for_constant() {
    let img = make_image(vec![std::f32::consts::PI; 64], [4, 4, 4]);
    let filt = PrewittFilter::unit();
    let out = filt.apply(&img).unwrap();
    let v = values(&out);
    let max_abs = v.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
    assert!(max_abs < 1e-5);
}

#[test]
fn prewitt_magnitude_isotropic_for_diagonal_ramp() {
    // I(z,y,x) = z + y + x => |∇I| = √3 ≈ 1.732
    let mut vals = Vec::with_capacity(27);
    for iz in 0..3 {
        for iy in 0..3 {
            for ix in 0..3 {
                vals.push((iz + iy + ix) as f32);
            }
        }
    }
    let img = make_image(vals, [3, 3, 3]);
    let filt = PrewittFilter::unit();
    let out = filt.apply(&img).unwrap();
    let v = values(&out);
    let interior_idx = 9 + 3 + 1;
    let expected = (3.0_f32).sqrt();
    assert!(
        (v[interior_idx] - expected).abs() < 1e-4,
        "diagonal magnitude should be √3 ≈ 1.732, got {}",
        v[interior_idx]
    );
}

#[test]
fn prewitt_anisotropic_spacing() {
    // I(x) = x with spacing 0.5 along x
    // True gradient dI/dx = 1/0.5 = 2.0
    let mut vals = Vec::with_capacity(27);
    for _iz in 0..3 {
        for _iy in 0..3 {
            for ix in 0..3 {
                vals.push(ix as f32);
            }
        }
    }
    let img = make_image(vals, [3, 3, 3]);
    let filt = PrewittFilter::new([1.0, 1.0, 0.5].into());
    let (_gz, _gy, gx) = filt.apply_components(&img).unwrap();
    let gx_v = values(&gx);
    let interior_idx = 9 + 3 + 1;
    assert!(
        (gx_v[interior_idx] - 2.0).abs() < 1e-5,
        "anisotropic gradient should be 2.0, got {}",
        gx_v[interior_idx]
    );
}

#[test]
fn prewitt_zero_orthogonal_components_for_axial_ramp() {
    // I(x) = x => gy = gz = 0
    let mut vals = Vec::with_capacity(27);
    for _iz in 0..3 {
        for _iy in 0..3 {
            for ix in 0..3 {
                vals.push(ix as f32);
            }
        }
    }
    let img = make_image(vals, [3, 3, 3]);
    let filt = PrewittFilter::unit();
    let (gz, gy, _gx) = filt.apply_components(&img).unwrap();
    let gz_v = values(&gz);
    let gy_v = values(&gy);
    let max_ortho = gz_v
        .iter()
        .chain(gy_v.iter())
        .fold(0.0_f32, |a, &b| a.max(b.abs()));
    assert!(
        max_ortho < 1e-5,
        "orthogonal components should be 0, got max |v| = {}",
        max_ortho
    );
}

#[test]
fn prewitt_preserves_shape() {
    let vals = vec![0.0_f32; 24];
    let img = make_image(vals, [2, 3, 4]);
    let filt = PrewittFilter::unit();
    let out = filt.apply(&img).unwrap();
    let v = values(&out);
    assert_eq!(v.len(), 24);
}

#[test]
fn prewitt_single_voxel_returns_zero() {
    // 1-voxel image: all axes degenerate, replicate padding returns self
    let img = make_image(vec![7.0_f32], [1, 1, 1]);
    let filt = PrewittFilter::unit();
    let out = filt.apply(&img).unwrap();
    let v = values(&out);
    assert!(
        v[0].abs() < 1e-5,
        "single voxel should yield 0, got {}",
        v[0]
    );
}
