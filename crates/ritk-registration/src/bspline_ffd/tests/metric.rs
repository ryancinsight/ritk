use super::super::metric::compute_ncc;

#[test]
fn ncc_identical_images_is_one() {
    let image: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
    let ncc = compute_ncc(&image, &image);
    assert!(
        (ncc - 1.0).abs() < 1e-10,
        "NCC of identical images should be 1.0, got {}",
        ncc
    );
}

#[test]
fn ncc_negated_image_is_minus_one() {
    let image: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1 - 5.0).collect();
    let neg: Vec<f32> = image.iter().map(|&v| -v).collect();
    let ncc = compute_ncc(&image, &neg);
    assert!(
        (ncc - (-1.0)).abs() < 1e-10,
        "NCC of negated images should be -1.0, got {}",
        ncc
    );
}
