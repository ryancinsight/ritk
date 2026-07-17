use super::NoiseImageFilter;
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = LegacyBurnBackend;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::burn_compat::make_image::<B, 3>(data, dims)
}

/// A constant image has zero local noise everywhere (variance is 0).
#[test]
fn local_noise_of_constant_is_zero() {
    let dims = [4usize, 5, 6];
    let n: usize = dims.iter().product();
    let out = NoiseImageFilter::default().apply(&make(vec![42.0; n], dims));
    let (ov, _) = extract_vec_infallible(&out);
    assert!(
        ov.iter().all(|&v| v == 0.0),
        "constant image must yield 0 noise"
    );
}

/// Interior value equals the closed-form sample standard deviation over the
/// 3×1×1 ZeroFluxNeumann window. With a 1-D ramp `I(x) = 10·x` and radius 1
/// along x only, the centre voxel x=2 sees {10, 20, 30}: mean 20, sample
/// variance `((10²+20²+30²) − 60²/3)/2 = 200/2... ` → sample sd = 10.
#[test]
fn local_noise_interior_matches_closed_form() {
    // A single-row volume so the y/z windows are degenerate (size 1, clamped),
    // isolating the x-axis sample standard deviation.
    let (nz, ny, nx) = (1usize, 1, 5);
    let vals: Vec<f32> = (0..nx).map(|x| 10.0 * x as f32).collect(); // 0,10,20,30,40
    let out = NoiseImageFilter::new([0, 0, 1]).apply(&make(vals, [nz, ny, nx]));
    let (ov, _) = extract_vec_infallible(&out);
    // Centre x=2 window {10,20,30}: mean 20, var=((100+400+900)-3600/3)/2=200,
    // sample sd = sqrt(200/2)? var=(1400-1200)/2=100 → sd=10.
    assert!(
        (ov[2] - 10.0).abs() < 1e-5,
        "interior sample sd: got {}, want 10",
        ov[2]
    );
    // x=0 ZeroFluxNeumann window {0,0,10} (left neighbour clamps to edge 0):
    // mean 10/3, var=((0+0+100)-(10²/3))/2=(100-33.333)/2=33.333 → sd=5.7735.
    assert!(
        (ov[0] - (100.0f32 / 3.0).sqrt()).abs() < 1e-4,
        "edge ZeroFluxNeumann sd: got {}, want {}",
        ov[0],
        (100.0f32 / 3.0).sqrt()
    );
}

/// Output geometry equals input geometry.
#[test]
fn local_noise_preserves_geometry() {
    let dims = [3usize, 4, 5];
    let n: usize = dims.iter().product();
    let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3).sin() * 20.0).collect();
    let img = make(vals, dims);
    let out = NoiseImageFilter::default().apply(&img);
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], img.spacing()[0]);
}
