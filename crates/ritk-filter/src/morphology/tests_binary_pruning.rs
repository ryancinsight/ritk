use super::BinaryPruningFilter;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}

/// A 2×2 block is stable: every pixel sees the other three (genus 3 ≥ 2), so no
/// pixel is ever pruned.
#[test]
fn binary_pruning_preserves_solid_block() {
    let (nz, ny, nx) = (1usize, 4, 4);
    let mut vals = vec![0.0f32; ny * nx];
    for (y, x) in [(1, 1), (1, 2), (2, 1), (2, 2)] {
        vals[y * nx + x] = 1.0;
    }
    let out = BinaryPruningFilter::default().apply(&make(vals.clone(), [nz, ny, nx]));
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(ov, vals, "a 2×2 block (genus 3) must survive pruning");
}

/// A 1-pixel-wide line fully vanishes in a single in-place sweep: pruning the
/// left endpoint drops its right neighbour to genus 1, which prunes in turn, and
/// the deletion cascades along the line — a direct check of the in-place,
/// raster-order semantics (a collect-then-delete sweep would leave the interior).
#[test]
fn binary_pruning_thin_line_cascades_away() {
    let (nz, ny, nx) = (1usize, 5, 9);
    let mut vals = vec![0.0f32; ny * nx];
    for x in 2..=6 {
        vals[2 * nx + x] = 1.0;
    }
    let out = BinaryPruningFilter::new(1).apply(&make(vals, [nz, ny, nx]));
    let (ov, _) = extract_vec_infallible(&out);
    assert!(
        ov.iter().all(|&v| v == 0.0),
        "an isolated 1-pixel line cascades to nothing in one sweep"
    );
}

/// An isolated pixel (genus 0) is always removed.
#[test]
fn binary_pruning_removes_isolated_pixel() {
    let (nz, ny, nx) = (1usize, 5, 5);
    let mut vals = vec![0.0f32; ny * nx];
    vals[2 * nx + 2] = 1.0;
    let out = BinaryPruningFilter::default().apply(&make(vals, [nz, ny, nx]));
    let (ov, _) = extract_vec_infallible(&out);
    assert!(
        ov.iter().all(|&v| v == 0.0),
        "isolated pixel must be pruned"
    );
}

/// Output is binary and geometry is preserved.
#[test]
fn binary_pruning_is_binary_and_preserves_geometry() {
    let dims = [1usize, 4, 6];
    let n: usize = dims.iter().product();
    let vals: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
    let img = make(vals, dims);
    let out = BinaryPruningFilter::default().apply(&img);
    let (ov, _) = extract_vec_infallible(&out);
    assert!(
        ov.iter().all(|&v| v == 0.0 || v == 1.0),
        "output must be binary"
    );
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], img.spacing()[0]);
}
