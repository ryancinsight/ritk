use super::*;
use crate::render::slice_render::RgbaImage;

/// Build a 3×2 image where pixel `(row, col)` holds `[row, col, 0, 255]`.
fn make_test_image() -> RgbaImage {
    RgbaImage::new(
        [3, 2],
        vec![
            0, 0, 0, 255, 0, 1, 0, 255, 0, 2, 0, 255, 1, 0, 0, 255, 1, 1, 0, 255, 1, 2, 0, 255,
        ],
    )
}

fn transformed(image: &RgbaImage, transform: ViewTransform) -> RgbaImage {
    let mut output = Vec::new();
    let size = apply_to_rgba_into(image.size(), image.pixels(), transform, &mut output)
        .expect("valid RGBA transform input");
    RgbaImage::new(size, output)
}

fn px(img: &RgbaImage, row: usize, col: usize) -> [u8; 4] {
    img.rgba_pixels()
        .nth(row * img.size()[0] + col)
        .expect("pixel within the image")
}

fn mirrored_left_to_right(image: &RgbaImage) -> RgbaImage {
    transformed(image, ViewTransform::default().toggle_flip_h())
}

fn mirrored_top_to_bottom(image: &RgbaImage) -> RgbaImage {
    transformed(image, ViewTransform::default().toggle_flip_v())
}

fn rotated_clockwise(image: &RgbaImage) -> RgbaImage {
    transformed(image, ViewTransform::default().rotate_cw())
}

// ── Identity ──────────────────────────────────────────────────────────────

#[test]
fn test_identity_no_change() {
    let img = make_test_image();
    let mut output = vec![7_u8; 3];
    let size = apply_to_rgba_into(
        img.size(),
        img.pixels(),
        ViewTransform::default(),
        &mut output,
    )
    .expect("valid RGBA transform input");
    assert_eq!(size, img.size(), "identity must preserve dimensions");
    assert_eq!(
        output,
        [7, 7, 7],
        "identity must leave the output untouched"
    );
}

// ── Flip horizontal ───────────────────────────────────────────────────────

#[test]
fn test_flip_h_pixel_mapping() {
    // Original: pixel(0,0)=[0,0,0, 255], pixel(0,1)=[0,1,0, 255]
    // After flip_h on width=3: col 0→2, col 1→1, col 2→0
    let img = make_test_image(); // 3×2
    let flipped = mirrored_left_to_right(&img);
    assert_eq!(flipped.size(), [3, 2], "flip_h must preserve dimensions");
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
    let twice = mirrored_left_to_right(&mirrored_left_to_right(&img));
    assert_eq!(twice.pixels(), img.pixels(), "flip_h ∘ flip_h = identity");
}

// ── Flip vertical ─────────────────────────────────────────────────────────

#[test]
fn test_flip_v_pixel_mapping() {
    let img = make_test_image(); // 3×2 (height=2)
    let flipped = mirrored_top_to_bottom(&img);
    assert_eq!(flipped.size(), [3, 2], "flip_v must preserve dimensions");
    // Row 0 ↔ row 1 (height = 2 → H-1-0 = 1).
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
    let twice = mirrored_top_to_bottom(&mirrored_top_to_bottom(&img));
    assert_eq!(twice.pixels(), img.pixels(), "flip_v ∘ flip_v = identity");
}

// ── Rotation ──────────────────────────────────────────────────────────────

#[test]
fn test_rotate_90_cw_dimensions() {
    // 3×2 (W=3, H=2) rotated 90° CW → 2×3 (W=H_orig, H=W_orig).
    let img = make_test_image();
    let r = rotated_clockwise(&img);
    assert_eq!(r.size(), [2, 3], "90° CW rotation: size must swap W↔H");
}

#[test]
fn test_rotate_90_cw_analytical_mapping() {
    // For 3×2 (W=3, H=2): pixel at (row=0, col=0) maps to output (orow=0, ocol=H-1-row=1).
    // For pixel at (row=1, col=2): output (orow=2, ocol=H-1-1=0).
    let img = make_test_image();
    let r = rotated_clockwise(&img);
    // Original (0,0) → output (0, 1)
    assert_eq!(px(&r, 0, 1), px(&img, 0, 0), "rotate_90_cw: (0,0)→(0,H-1)");
    // Original (0,2) → output (2, 1)
    assert_eq!(
        px(&r, 2, 1),
        px(&img, 0, 2),
        "rotate_90_cw: (0,W-1)→(W-1,H-1)"
    );
    // Original (1,0) → output (0, 0)
    assert_eq!(px(&r, 0, 0), px(&img, 1, 0), "rotate_90_cw: (H-1,0)→(0,0)");
}

#[test]
fn test_four_rotations_is_identity() {
    // Four 90° CW rotations must produce the original dimensions and pixel values.
    let img = make_test_image();
    let mut r = img.clone();
    for _ in 0..4 {
        r = rotated_clockwise(&r);
    }
    assert_eq!(
        r.size(),
        img.size(),
        "4×90° rotation: dimensions must be restored"
    );
    assert_eq!(
        r.pixels(),
        img.pixels(),
        "4×90° rotation: pixels must be restored"
    );
}

// ── ViewTransform helpers ─────────────────────────────────────────────────

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

#[test]
fn test_continuous_point_mapping_round_trips_all_transforms() {
    let source_size = [7, 4];
    let points = [[0.0, 0.0], [7.0, 0.0], [0.0, 4.0], [7.0, 4.0], [2.25, 1.75]];

    for transform in all_transforms() {
        let quarter_turn = matches!(
            transform.rotation,
            RotationSteps::Ninety | RotationSteps::TwoSeventy
        );
        let (expected, limits) = if quarter_turn {
            ([4, 7], [4.0, 7.0])
        } else {
            (source_size, [7.0, 4.0])
        };
        assert_eq!(
            transform.output_size(source_size),
            expected,
            "output dimensions for {transform:?}"
        );
        // The points and extents are dyadic, so f64 reflection is exact.
        for point in points {
            let mapped = transform.source_to_output_coordinates(point, source_size);
            assert_eq!(
                transform.output_to_source_coordinates(mapped, source_size),
                point
            );
            assert!((0.0..=limits[0]).contains(&mapped[0]));
            assert!((0.0..=limits[1]).contains(&mapped[1]));
        }
    }
}

#[test]
fn test_continuous_point_mapping_matches_pixel_centres() {
    let source_size = [3, 2];
    let source = [0.5, 0.5];
    let transform = ViewTransform {
        flip_h: true,
        flip_v: false,
        rotation: RotationSteps::Ninety,
    };
    let output = transform.source_to_output_coordinates(source, source_size);
    assert_eq!(output, [1.5, 2.5]);
    assert_eq!(
        transform.output_to_source_coordinates(output, source_size),
        source
    );
}

// ── Reusable storage ───────────────────────────────────────────────────
//
// Every (flip_h, flip_v, rotation) combination (2×2×4 = 16) places pixels
// where the display coordinate mapping says they are, and reuses its storage.

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
fn test_apply_to_rgba_into_agrees_with_the_display_mapping() {
    let img = make_test_image();
    let mut output = Vec::new();
    for transform in all_transforms().into_iter().filter(|t| !t.is_identity()) {
        let size = apply_to_rgba_into(img.size(), img.pixels(), transform, &mut output)
            .expect("valid RGBA transform input");
        for (row, line) in (0_u8..).zip(output.chunks_exact(size[0] * 4)) {
            for (column, pixel) in (0_u8..).zip(line.chunks_exact(4)) {
                let centre = [f64::from(column) + 0.5, f64::from(row) + 0.5];
                let source = transform.output_to_source_coordinates(centre, img.size());
                // Each pixel stores its source row and column. Half-integers
                // this small are exact, so the mapping is compared exactly.
                assert_eq!(
                    source,
                    [f64::from(pixel[1]) + 0.5, f64::from(pixel[0]) + 0.5],
                    "{transform:?} at output row {row}, column {column}"
                );
            }
        }
    }
}

#[test]
fn test_apply_to_rgba_into_reuses_storage_for_every_transform() {
    let img = make_test_image();
    let mut output = Vec::new();
    for transform in all_transforms() {
        let size = apply_to_rgba_into(img.size(), img.pixels(), transform, &mut output)
            .expect("valid reusable RGBA transform input");
        assert_eq!(size, transform.output_size(img.size()));
        if !transform.is_identity() {
            assert_eq!(output.len(), size[0] * size[1] * 4);
        }
    }
    let warm_capacity = output.capacity();
    for transform in all_transforms() {
        apply_to_rgba_into(img.size(), img.pixels(), transform, &mut output)
            .expect("repeat reusable RGBA transform input");
        assert_eq!(output.capacity(), warm_capacity);
    }
}
