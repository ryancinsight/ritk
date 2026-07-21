use super::ZeroCrossingBasedEdgeDetectionFilter;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}

/// The output is binary: every voxel is either the foreground (edge) or the
/// background value, matching ITK's `ZeroCrossingImageFilter` labelling.
#[test]
fn zero_crossing_edge_output_is_binary() {
    let (nz, ny, nx) = (6usize, 8, 8);
    let n = nz * ny * nx;
    // A step edge along x: a sign change the Laplacian/zero-crossing will mark.
    let mut vals = vec![0.0f32; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                vals[iz * ny * nx + iy * nx + ix] = if ix < nx / 2 { 0.0 } else { 100.0 };
            }
        }
    }
    let out = ZeroCrossingBasedEdgeDetectionFilter::default().apply(&make(vals, [nz, ny, nx]));
    let (ov, _) = extract_vec_infallible(&out.expect("infallible: validated precondition"));
    assert!(
        ov.iter().all(|&v| v == 0.0 || v == 1.0),
        "output must be binary {{0, 1}}"
    );
    // A step edge must produce at least one edge voxel.
    assert!(ov.contains(&1.0), "expected detected edge voxels");
}

/// Custom foreground/background labels are honoured.
#[test]
fn zero_crossing_edge_custom_labels() {
    let (nz, ny, nx) = (5usize, 6, 6);
    let n = nz * ny * nx;
    let mut vals = vec![0.0f32; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                vals[iz * ny * nx + iy * nx + ix] = if iy < ny / 2 { -50.0 } else { 50.0 };
            }
        }
    }
    let f = ZeroCrossingBasedEdgeDetectionFilter::new(1.0, 0.01, 7.0, 3.0);
    let out = f
        .apply(&make(vals, [nz, ny, nx]))
        .expect("infallible: validated precondition");
    let (ov, _) = extract_vec_infallible(&out);
    assert!(
        ov.iter().all(|&v| v == 7.0 || v == 3.0),
        "output must use the custom labels {{7, 3}}"
    );
}

/// Output geometry equals input geometry.
#[test]
fn zero_crossing_edge_preserves_geometry() {
    let dims = [4usize, 5, 6];
    let n: usize = dims.iter().product();
    let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).sin() * 30.0).collect();
    let img = make(vals, dims);
    let out = ZeroCrossingBasedEdgeDetectionFilter::default()
        .apply(&img)
        .expect("infallible: validated precondition");
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], img.spacing()[0]);
}
