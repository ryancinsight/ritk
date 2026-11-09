use super::*;

/// The spacing ratio, the two texel products and the frame product round each
/// extent, and the two ratios compared round once each: ten unit roundoffs, five
/// epsilon to first order. Eight epsilon covers the higher-order terms.
const ASPECT_TOLERANCE: f64 = 8.0 * f64::EPSILON;

/// The extent cancels from a panel centre, so only the subtraction, the origin
/// sum and the centre sum round, each by at most half an epsilon of a value below
/// 820 pixels. The final subtraction of nearby values is exact.
const POSITION_TOLERANCE: f64 = 3.0 * 820.0 * 0.5 * f64::EPSILON;

fn spacing(row: f64, column: f64) -> PresentationSpacing {
    PresentationSpacing::try_new(row, column).expect("positive finite sample distances")
}

fn no_pan() -> ViewportOffset {
    ViewportOffset::new(0.0, 0.0)
}

/// A 3 x 2 x 4 voxel study sampled at `[2, 1.5, 0.5]` mm, one case per axis:
/// frame size, `[row, column]` spacing and the physical width over height.
const PLANES: [([u32; 2], [f64; 2], f64); 3] = [
    ([4, 2], [1.5, 0.5], 2.0 / 3.0),
    ([4, 3], [2.0, 0.5], 1.0 / 3.0),
    ([2, 3], [2.0, 1.5], 1.0 / 2.0),
];

#[test]
fn every_plane_keeps_its_physical_proportions_in_every_panel_shape() {
    for (frame, [row, column], expected) in PLANES {
        for (width, height) in [(800, 600), (120, 900), (1000, 100), (257, 263)] {
            for zoom in [1.0, 3.25] {
                let rect = placement_geometry(
                    frame,
                    spacing(row, column),
                    0,
                    0,
                    width,
                    height,
                    zoom,
                    no_pan(),
                )
                .expect("representable placement");
                assert!(
                    (rect.width / rect.height - expected).abs() <= ASPECT_TOLERANCE * expected,
                    "{frame:?} in {width}x{height} at zoom {zoom}: {rect:?}"
                );
            }
        }
    }
}

#[test]
fn a_common_distance_scale_does_not_change_the_placement() {
    // A power of two scales every distance exactly, so no rounding separates the fits.
    let scale = 2.0_f64.powi(100);
    let unit = placement_geometry([4, 2], spacing(1.5, 0.5), 0, 0, 800, 600, 1.0, no_pan())
        .expect("representable placement");
    let scaled = placement_geometry(
        [4, 2],
        spacing(1.5 * scale, 0.5 * scale),
        0,
        0,
        800,
        600,
        1.0,
        no_pan(),
    )
    .expect("a common scale cancels from the fit");
    assert_eq!(unit.width, scaled.width);
    assert_eq!(unit.height, scaled.height);
}

/// Each fill fraction carries the rounding of the two physical extents, the
/// fit, the texel and frame products and its own quotient, inside the ten unit
/// roundoffs `ASPECT_TOLERANCE` bounds.
#[test]
fn the_fit_fills_the_tighter_panel_dimension_and_stays_inside_the_other() {
    for (width, height) in [(800, 600), (120, 900), (1000, 100)] {
        let rect = placement_geometry(
            [4, 2],
            spacing(1.5, 0.5),
            0,
            0,
            width,
            height,
            1.0,
            no_pan(),
        )
        .expect("representable placement");
        let width_fill = rect.width / f64::from(width);
        let height_fill = rect.height / f64::from(height);
        assert!(
            (width_fill.max(height_fill) - 1.0).abs() <= ASPECT_TOLERANCE,
            "{width}x{height}: {rect:?}"
        );
    }
}

#[test]
fn the_fitted_image_is_centred_in_its_panel_and_pan_moves_it() {
    let centred = placement_geometry([4, 2], spacing(1.5, 0.5), 10, 20, 800, 600, 1.0, no_pan())
        .expect("representable placement");
    assert!((centred.x + centred.width * 0.5 - 410.0).abs() <= POSITION_TOLERANCE);
    assert!((centred.y + centred.height * 0.5 - 320.0).abs() <= POSITION_TOLERANCE);

    let panned = placement_geometry(
        [4, 2],
        spacing(1.5, 0.5),
        10,
        20,
        800,
        600,
        1.0,
        ViewportOffset::new(24.0, -8.0),
    )
    .expect("representable placement");
    assert!((panned.x - centred.x - 24.0).abs() <= POSITION_TOLERANCE);
    assert!((panned.y - centred.y + 8.0).abs() <= POSITION_TOLERANCE);
}

#[test]
fn geometry_beyond_the_host_range_is_rejected() {
    let error = placement_geometry(
        [1, 1],
        spacing(1.0, 1.0),
        0,
        0,
        u32::MAX,
        u32::MAX,
        f32::MAX,
        no_pan(),
    )
    .expect_err("the texel size exceeds the host range");
    assert!(error.to_string().contains("finite host range"), "{error}");
}
