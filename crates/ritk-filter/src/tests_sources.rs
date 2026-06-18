use super::{gaussian_image_source, grid_image_source};

/// Peak at the mean voxel, value = scale; symmetric falloff matching
/// `exp(−½·((p−mean)/sigma)²)`. 1-D along x: size 9, sigma 6, mean 8, scale 100,
/// origin 2, spacing 1 → physical p = 2+x, peak at x=6 (p=8). Pinned by sitk.
#[test]
fn gaussian_source_matches_physical_formula() {
    let (buf, dims) = gaussian_image_source(
        [9, 1, 1],
        [6.0, 1.0, 1.0],
        [8.0, 0.0, 0.0],
        100.0,
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    );
    assert_eq!(dims, [1, 1, 9]);
    // Peak at x=6 (physical 8 == mean).
    assert_eq!(buf.iter().cloned().fold(f32::MIN, f32::max), buf[6]);
    assert!((buf[6] - 100.0).abs() < 1e-3, "peak {}", buf[6]);
    for (x, &got) in buf.iter().enumerate() {
        let p = 2.0 + x as f64;
        let exp = 100.0 * (-0.5 * ((p - 8.0) / 6.0_f64).powi(2)).exp();
        assert!((got as f64 - exp).abs() < 1e-3, "x={x}: {got} vs {exp}");
    }
}

/// 3-D separability: the value at the centre voxel equals the product of the
/// per-axis Gaussians (a constant 1 here since each axis peaks at its mean).
#[test]
fn gaussian_source_3d_peak_is_scale() {
    let (buf, dims) = gaussian_image_source(
        [5, 5, 5],
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0], // mean at index 2 each axis (origin 0, spacing 1)
        50.0,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    );
    assert_eq!(dims, [5, 5, 5]);
    let center = (2 * 5 + 2) * 5 + 2;
    assert!((buf[center] - 50.0).abs() < 1e-3, "center {}", buf[center]);
}

/// 1-D grid: dark on grid lines (x=0,4,8), bright between. Pinned by sitk:
/// `gridSpacing=4, sigma=0.5, scale=255` → x=0→0, x=1→220.49, x=2→254.83.
#[test]
fn grid_source_dark_lines_bright_between() {
    let (buf, dims) = grid_image_source(
        [12, 1, 1],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [4.0, 4.0, 4.0],
        [0.0, 0.0, 0.0],
        255.0,
        [true, false, false],
    );
    assert_eq!(dims, [1, 1, 12]);
    assert!(buf[0].abs() < 1e-2, "line x=0 should be ~0, got {}", buf[0]);
    assert!((buf[1] - 220.49).abs() < 0.1, "x=1: {}", buf[1]);
    assert!((buf[2] - 254.83).abs() < 0.1, "x=2: {}", buf[2]);
    assert!(buf[4].abs() < 1e-2, "line x=4 should be ~0, got {}", buf[4]);
}
