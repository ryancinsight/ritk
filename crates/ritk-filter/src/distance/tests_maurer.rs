use super::SignedMaurerDistanceMapImageFilter;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_tensor_ops::extract_vec;

type B = NdArray<f32>;

/// 3×3 foreground block in a 9×9 image — the canonical ITK Maurer reference.
/// Border voxels (the whole 3×3 ring, all 8-adjacent to background) are the
/// distance sources, so the centre sits at distance 1 (→ −1), the ring at 0,
/// and the far corner at √18 = 4.2426 (→ +4.2426). These exact values were
/// confirmed against `sitk.SignedMaurerDistanceMap`.
#[test]
fn test_signed_maurer_3x3_block_values() {
    let (ny, nx) = (9usize, 9);
    let mut img = vec![0.0f32; ny * nx];
    for y in 3..6 {
        for x in 3..6 {
            img[y * nx + x] = 1.0;
        }
    }
    let out = SignedMaurerDistanceMapImageFilter {
        squared_distance: false,
        ..Default::default()
    }
    .apply(&ts::make_image::<B, 3>(img, [1, ny, nx]))
    .unwrap();
    let (d, _) = extract_vec(&out).unwrap();
    let at = |y: usize, x: usize| d[y * nx + x];

    // Foreground centre: nearest border voxel at distance 1 → −1.
    assert!((at(4, 4) - (-1.0)).abs() < 1e-5, "centre = {}", at(4, 4));
    // Border ring voxel: on the border → 0.
    assert!(at(3, 3).abs() < 1e-5, "ring = {}", at(3, 3));
    // Background adjacent to the block face: +1.
    assert!((at(3, 2) - 1.0).abs() < 1e-5, "face bg = {}", at(3, 2));
    // Far corner: √18 = 4.242641.
    assert!((at(0, 0) - 18.0_f32.sqrt()).abs() < 1e-5, "corner = {}", at(0, 0));
}

/// Squared-distance mode returns the signed square of the distance.
#[test]
fn test_signed_maurer_squared() {
    let (ny, nx) = (9usize, 9);
    let mut img = vec![0.0f32; ny * nx];
    for y in 3..6 {
        for x in 3..6 {
            img[y * nx + x] = 1.0;
        }
    }
    let out = SignedMaurerDistanceMapImageFilter {
        squared_distance: true,
        ..Default::default()
    }
    .apply(&ts::make_image::<B, 3>(img, [1, ny, nx]))
    .unwrap();
    let (d, _) = extract_vec(&out).unwrap();
    // Far corner squared: 18; sign positive (background).
    assert!((d[0] - 18.0).abs() < 1e-4, "corner² = {}", d[0]);
    // Centre squared: 1; sign negative (foreground).
    assert!((d[4 * nx + 4] - (-1.0)).abs() < 1e-5, "centre² = {}", d[4 * nx + 4]);
}

/// `inside_is_positive = true` flips the sign convention.
#[test]
fn test_signed_maurer_inside_positive() {
    let (ny, nx) = (9usize, 9);
    let mut img = vec![0.0f32; ny * nx];
    for y in 3..6 {
        for x in 3..6 {
            img[y * nx + x] = 1.0;
        }
    }
    let out = SignedMaurerDistanceMapImageFilter {
        squared_distance: false,
        inside_is_positive: true,
        ..Default::default()
    }
    .apply(&ts::make_image::<B, 3>(img, [1, ny, nx]))
    .unwrap();
    let (d, _) = extract_vec(&out).unwrap();
    // Foreground centre now positive.
    assert!((d[4 * nx + 4] - 1.0).abs() < 1e-5, "centre = {}", d[4 * nx + 4]);
    // Background corner now negative.
    assert!((d[0] - (-(18.0_f32.sqrt()))).abs() < 1e-5, "corner = {}", d[0]);
}
