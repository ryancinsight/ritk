use super::FastMarchingFilter;
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = LegacyBurnBackend;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}

/// On a 1-D unit-speed line with one seed, the upwind quadratic reduces to
/// `T(x) = T(neighbour) + 1`, so arrival time equals the index distance to the
/// seed.
#[test]
fn fast_marching_1d_unit_speed_is_distance() {
    let nx = 7usize;
    let speed = make(vec![1.0f32; nx], [1, 1, nx]);
    let out = FastMarchingFilter::new(vec![[0, 0, 3]]).apply(&speed);
    let (ov, _) = extract_vec_infallible(&out);
    let want = [3.0f32, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0];
    for (i, (&got, &w)) in ov.iter().zip(want.iter()).enumerate() {
        assert!((got - w).abs() < 1e-5, "voxel {i}: got {got}, want {w}");
    }
}

/// Halving the speed doubles the arrival time (the Eikonal equation scales the
/// time by `1/F`).
#[test]
fn fast_marching_speed_scales_arrival_time() {
    let nx = 6usize;
    let out_fast =
        FastMarchingFilter::new(vec![[0, 0, 0]]).apply(&make(vec![1.0f32; nx], [1, 1, nx]));
    let out_slow =
        FastMarchingFilter::new(vec![[0, 0, 0]]).apply(&make(vec![0.5f32; nx], [1, 1, nx]));
    let (f, _) = extract_vec_infallible(&out_fast);
    let (s, _) = extract_vec_infallible(&out_slow);
    for i in 0..nx {
        assert!(
            (s[i] - 2.0 * f[i]).abs() < 1e-4,
            "voxel {i}: slow {} vs 2·fast {}",
            s[i],
            f[i]
        );
    }
}

/// The seed has zero arrival time and output geometry is preserved.
#[test]
fn fast_marching_seed_zero_and_geometry() {
    let dims = [1usize, 5, 6];
    let n: usize = dims.iter().product();
    let out = FastMarchingFilter::new(vec![[0, 2, 3]]).apply(&make(vec![1.0; n], dims));
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(ov[2 * 6 + 3], 0.0, "seed arrival time is 0");
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], 1.0);
}
