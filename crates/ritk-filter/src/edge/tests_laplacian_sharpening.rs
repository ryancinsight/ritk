use super::LaplacianSharpeningFilter;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}

/// The output is clamped to the input intensity range `[min I, max I]` (ITK's
/// final `std::clamp`), so no output voxel may leave that interval.
#[test]
fn laplacian_sharpening_clamps_to_input_range() {
    let (nz, ny, nx) = (6usize, 7, 8);
    let n = nz * ny * nx;
    let mut vals = vec![0.0f32; n];
    for (i, v) in vals.iter_mut().enumerate() {
        *v = ((i as f32 * 0.3).sin() * 40.0 + 50.0).floor();
    }
    let (lo, hi) = vals
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let out = LaplacianSharpeningFilter::default().apply(&make(vals, [nz, ny, nx]));
    let (ov, _) = extract_vec_infallible(&out);
    for (i, &v) in ov.iter().enumerate() {
        assert!(
            v >= lo - 1e-4 && v <= hi + 1e-4,
            "voxel {i}={v} left input range [{lo}, {hi}]"
        );
    }
}

/// Sharpening a textured image with curvature is not the identity: interior
/// voxels away from the clamp bounds change under the Laplacian subtraction.
#[test]
fn laplacian_sharpening_is_not_identity() {
    let (nz, ny, nx) = (6usize, 6, 6);
    let n = nz * ny * nx;
    let mut vals = vec![0.0f32; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                // Smooth multi-axis texture with real curvature everywhere.
                vals[iz * ny * nx + iy * nx + ix] =
                    ((ix as f32 * 0.9).sin() + (iy as f32 * 0.7).cos() + (iz as f32 * 1.1).sin())
                        * 25.0
                        + 60.0;
            }
        }
    }
    let inp = vals.clone();
    let out = LaplacianSharpeningFilter::default().apply(&make(vals, [nz, ny, nx]));
    let (ov, _) = extract_vec_infallible(&out);
    let max_change = ov
        .iter()
        .zip(inp.iter())
        .map(|(&o, &i)| (o - i).abs())
        .fold(0.0f32, f32::max);
    assert!(max_change > 1e-3, "sharpening left the image unchanged");
}

/// Output geometry equals input geometry.
#[test]
fn laplacian_sharpening_preserves_geometry() {
    let dims = [4usize, 5, 6];
    let n: usize = dims.iter().product();
    let vals: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.7).cos() * 20.0 + 30.0)
        .collect();
    let img = make(vals, dims);
    let out = LaplacianSharpeningFilter::new(false).apply(&img);
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], img.spacing()[0]);
}
