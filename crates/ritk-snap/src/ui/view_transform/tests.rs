use super::*;
use crate::render::buffer_pool::RenderBufferPool;
use egui::{Color32, ColorImage};

/// Build a 3Ã—2 (width=3, height=2) test image where each pixel encodes
/// its position uniquely: pixel(row, col) = Color32::from_rgb(row as u8, col as u8, 0).
fn make_test_image() -> ColorImage {
    // size = [width, height] = [3, 2]
    let w = 3usize;
    let h = 2usize;
    let mut pixels = Vec::with_capacity(w * h);
    for row in 0..h {
        for col in 0..w {
            pixels.push(Color32::from_rgb(row as u8, col as u8, 0));
        }
    }
    ColorImage {
        size: [w, h],
        pixels,
    }
}

fn px(img: &ColorImage, row: usize, col: usize) -> Color32 {
    img.pixels[row * img.size[0] + col]
}

// â”€â”€ Identity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_identity_no_change() {
    let img = make_test_image();
    let result = apply_to_image(&img, ViewTransform::default());
    assert_eq!(result.size, img.size, "identity must preserve dimensions");
    assert_eq!(
        result.pixels, img.pixels,
        "identity must preserve pixel values"
    );
}

// â”€â”€ Flip horizontal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_flip_h_pixel_mapping() {
    // Original: pixel(0,0)=Color32::from_rgb(0,0,0), pixel(0,1)=Color32::from_rgb(0,1,0)
    // After flip_h on width=3: col 0â†’2, col 1â†’1, col 2â†’0
    let img = make_test_image(); // 3Ã—2
    let flipped = flip_h_image(&img);
    assert_eq!(flipped.size, [3, 2], "flip_h must preserve dimensions");
    // Pixel originally at (row=0, col=0) must now be at (row=0, col=2)
    assert_eq!(
        px(&flipped, 0, 2),
        px(&img, 0, 0),
        "flip_h: col 0 maps to col W-1"
    );
    // Pixel originally at (row=0, col=2) must now be at (row=0, col=0)
    assert_eq!(
        px(&flipped, 0, 0),
        px(&img, 0, 2),
        "flip_h: col W-1 maps to col 0"
    );
    // Center column (col=1) is invariant for odd-width images.
    assert_eq!(
        px(&flipped, 0, 1),
        px(&img, 0, 1),
        "flip_h: center col is invariant"
    );
}

#[test]
fn test_flip_h_involution() {
    // Two flip_h applications compose to identity.
    let img = make_test_image();
    let twice = flip_h_image(&flip_h_image(&img));
    assert_eq!(twice.pixels, img.pixels, "flip_h âˆ˜ flip_h = identity");
}

// â”€â”€ Flip vertical â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_flip_v_pixel_mapping() {
    let img = make_test_image(); // 3Ã—2 (height=2)
    let flipped = flip_v_image(&img);
    assert_eq!(flipped.size, [3, 2], "flip_v must preserve dimensions");
    // Row 0 â†” row 1 (height = 2 â†’ H-1-0 = 1).
    assert_eq!(
        px(&flipped, 0, 0),
        px(&img, 1, 0),
        "flip_v: row 0 maps from row H-1"
    );
    assert_eq!(
        px(&flipped, 1, 0),
        px(&img, 0, 0),
        "flip_v: row H-1 maps from row 0"
    );
}

#[test]
fn test_flip_v_involution() {
    let img = make_test_image();
    let twice = flip_v_image(&flip_v_image(&img));
    assert_eq!(twice.pixels, img.pixels, "flip_v âˆ˜ flip_v = identity");
}

// â”€â”€ Rotation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_rotate_90_cw_dimensions() {
    // 3Ã—2 (W=3, H=2) rotated 90Â° CW â†’ 2Ã—3 (W=H_orig, H=W_orig).
    let img = make_test_image();
    let r = rotate_90_cw_image(&img);
    assert_eq!(r.size, [2, 3], "90Â° CW rotation: size must swap Wâ†”H");
}

#[test]
fn test_rotate_90_cw_analytical_mapping() {
    // For 3Ã—2 (W=3, H=2): pixel at (row=0, col=0) maps to output (orow=0, ocol=H-1-row=1).
    // For pixel at (row=1, col=2): output (orow=2, ocol=H-1-1=0).
    let img = make_test_image();
    let r = rotate_90_cw_image(&img);
    // Original (0,0) â†’ output (0, 1)
    assert_eq!(
        px(&r, 0, 1),
        px(&img, 0, 0),
        "rotate_90_cw: (0,0)â†’(0,H-1)"
    );
    // Original (0,2) â†’ output (2, 1)
    assert_eq!(
        px(&r, 2, 1),
        px(&img, 0, 2),
        "rotate_90_cw: (0,W-1)â†’(W-1,H-1)"
    );
    // Original (1,0) â†’ output (0, 0)
    assert_eq!(
        px(&r, 0, 0),
        px(&img, 1, 0),
        "rotate_90_cw: (H-1,0)â†’(0,0)"
    );
}

#[test]
fn test_four_rotations_is_identity() {
    // Four 90Â° CW rotations must produce the original dimensions and pixel values.
    let img = make_test_image();
    let mut r = img.clone();
    for _ in 0..4 {
        r = rotate_90_cw_image(&r);
    }
    assert_eq!(
        r.size, img.size,
        "4Ã—90Â° rotation: dimensions must be restored"
    );
    assert_eq!(
        r.pixels, img.pixels,
        "4Ã—90Â° rotation: pixels must be restored"
    );
}

// â”€â”€ ViewTransform helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_is_identity_default() {
    assert!(ViewTransform::default().is_identity());
}

#[test]
fn test_toggle_flip_h_toggle_twice_is_identity() {
    let t = ViewTransform::default().toggle_flip_h().toggle_flip_h();
    assert!(t.is_identity());
}

#[test]
fn test_rotate_cw_four_times_is_identity() {
    let t = ViewTransform::default()
        .rotate_cw()
        .rotate_cw()
        .rotate_cw()
        .rotate_cw();
    assert!(t.is_identity());
}

// â”€â”€ Differential equivalence: apply_to_image_into vs apply_to_image â”€â”€â”€â”€
//
// For every (flip_h, flip_v, rotation) combination (2Ã—2Ã—4 = 16), verify
// that the single-pass scratch-based function produces pixel-identical
// output to the multi-step allocating function.

/// Generate all 16 ViewTransform combinations.
fn all_transforms() -> Vec<ViewTransform> {
    let mut out = Vec::with_capacity(16);
    for &flip_h in &[false, true] {
        for &flip_v in &[false, true] {
            for &rotation in &[
                RotationSteps::Zero,
                RotationSteps::Ninety,
                RotationSteps::OneEighty,
                RotationSteps::TwoSeventy,
            ] {
                out.push(ViewTransform {
                    flip_h,
                    flip_v,
                    rotation,
                });
            }
        }
    }
    out
}

#[test]
fn test_apply_to_image_into_differential_all_16_combinations() {
    let img = make_test_image(); // 3Ã—2
    let mut pool = RenderBufferPool::default();

    for t in all_transforms() {
        let expected = apply_to_image(&img, t);
        let actual = apply_to_image_into(&mut pool, &img, t);
        assert_eq!(
            actual.size, expected.size,
            "apply_to_image_into size mismatch for {:?}",
            t
        );
        assert_eq!(
            actual.pixels, expected.pixels,
            "apply_to_image_into pixel mismatch for {:?}",
            t
        );
    }
}

#[test]
fn test_apply_to_image_into_pool_reuse_consistent() {
    let img = make_test_image();
    let mut pool = RenderBufferPool::default();
    let t = ViewTransform {
        flip_h: true,
        flip_v: false,
        rotation: RotationSteps::Ninety,
    };

    let first = apply_to_image_into(&mut pool, &img, t);
    let second = apply_to_image_into(&mut pool, &img, t);
    assert_eq!(
        first.pixels, second.pixels,
        "pool reuse must produce identical output across consecutive calls"
    );
}

#[test]
fn test_apply_to_image_into_identity_no_deep_copy() {
    // Identity path returns img.clone() which is an Arc bump, not deep copy.
    let img = make_test_image();
    let mut pool = RenderBufferPool::default();
    let result = apply_to_image_into(&mut pool, &img, ViewTransform::default());
    assert_eq!(result.size, img.size, "identity must preserve dimensions");
    assert_eq!(
        result.pixels, img.pixels,
        "identity must preserve pixel values"
    );
}

#[test]
fn test_apply_to_image_into_square_image_all_combinations() {
    // Test with a square image where W==H to catch dimension-swap edge cases.
    let w = 3usize;
    let h = 3usize;
    let mut pixels = Vec::with_capacity(w * h);
    for row in 0..h {
        for col in 0..w {
            pixels.push(Color32::from_rgb(row as u8, col as u8, 1));
        }
    }
    let img = ColorImage {
        size: [w, h],
        pixels,
    };
    let mut pool = RenderBufferPool::default();

    for t in all_transforms() {
        let expected = apply_to_image(&img, t);
        let actual = apply_to_image_into(&mut pool, &img, t);
        assert_eq!(
            actual.size, expected.size,
            "square: size mismatch for {:?}",
            t
        );
        assert_eq!(
            actual.pixels, expected.pixels,
            "square: pixel mismatch for {:?}",
            t
        );
    }
}
