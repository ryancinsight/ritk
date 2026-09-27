use super::{ObliqueViewport, ObliqueViewportError};
use crate::geometry::PatientPointMm;
use crate::presentation::ViewportPoint;
use crate::render::{ResliceInterpolation, ReslicePlane};
use crate::LoadedVolume;

#[test]
fn pixel_centres_map_to_reslice_coordinates() {
    let viewport =
        ObliqueViewport::try_new([10.0, 20.0], [2.0, 4.0], [5, 3], [10.0, 20.0, 20.0, 32.0])
            .expect("finite non-empty image geometry");
    assert_eq!(
        viewport.map(ViewportPoint::new(17.0, 30.0)),
        Some([3.0, 2.0])
    );
    assert_eq!(
        viewport.map(ViewportPoint::new(11.0, 22.0)),
        Some([0.0, 0.0])
    );
    assert_eq!(
        viewport.map(ViewportPoint::new(19.9, 31.9)),
        Some([4.0, 2.0])
    );
    assert_eq!(
        viewport.map(ViewportPoint::new(16.0, 28.0)),
        Some([2.5, 1.5])
    );
}

#[test]
fn interior_point_rounding_to_the_image_edge_maps_to_last_pixel() {
    let viewport =
        ObliqueViewport::try_new([-1.0, 0.0], [0.1, 1.0], [10, 1], [-1.0, 0.0, 1.0, 1.0])
            .expect("finite image rectangle with a sub-ulp far edge");

    assert_eq!(viewport.map(ViewportPoint::new(0.0, 0.5)), Some([9.0, 0.0]));
}

#[test]
fn resized_zoomed_and_panned_rectangles_preserve_pixel_mapping() {
    let image_origin = [10.0, 20.0];
    let zoom_anchor = [20.0, 30.0];
    let zoom = 1.5;
    let pan = [7.0, -5.0];
    let transformed_origin = std::array::from_fn(|axis| {
        zoom_anchor[axis] + pan[axis] + (image_origin[axis] - zoom_anchor[axis]) * zoom
    });
    let frame = [5, 3];
    let source_pixel = [3.0, 1.0];

    for (origin, texel_size, pane, expected) in [
        (
            image_origin,
            [2.0, 4.0],
            [0.0, 0.0, 200.0, 200.0],
            [17.0, 26.0],
        ),
        (
            transformed_origin,
            [3.0, 6.0],
            [0.0, 0.0, 200.0, 200.0],
            [22.5, 19.0],
        ),
        (
            [100.0, 50.0],
            [2.5, 5.0],
            [90.0, 40.0, 150.0, 80.0],
            [108.75, 57.5],
        ),
    ] {
        let viewport = ObliqueViewport::try_new(origin, texel_size, frame, pane)
            .expect("resized, zoomed and panned image geometry");
        let screen = viewport
            .screen_point(source_pixel)
            .expect("source pixel remains visible in the image rectangle");
        assert_eq!(screen, expected);
        assert_eq!(
            viewport.map(ViewportPoint::new(screen[0], screen[1])),
            Some(source_pixel)
        );
    }
}

#[test]
fn pointer_mapping_rejects_padding_clipping_and_non_finite_positions() {
    let viewport =
        ObliqueViewport::try_new([8.0, 18.0], [2.0, 4.0], [5, 3], [10.0, 20.0, 18.0, 30.0])
            .expect("finite image extending beyond its pane");
    assert_eq!(viewport.map(ViewportPoint::new(9.0, 22.0)), None);
    assert_eq!(viewport.map(ViewportPoint::new(17.0, 19.0)), None);
    assert_eq!(viewport.map(ViewportPoint::new(18.0, 22.0)), None);
    assert_eq!(viewport.map(ViewportPoint::new(12.0, f64::NAN)), None);
    assert_eq!(
        viewport.visible_pixel_bounds(),
        Some([10.0, 17.0, 20.0, 29.0])
    );

    let outside =
        ObliqueViewport::try_new([20.0, 20.0], [1.0, 1.0], [4, 4], [0.0, 0.0, 10.0, 10.0])
            .expect("valid image geometry outside pane");
    assert_eq!(outside.visible_pixel_bounds(), None);
    assert_eq!(outside.map(ViewportPoint::new(20.5, 20.5)), None);
}

#[test]
fn invalid_or_unrepresentable_geometry_is_rejected() {
    assert_eq!(
        ObliqueViewport::try_new([0.0, 0.0], [1.0, 1.0], [0, 1], [0.0, 0.0, 1.0, 1.0]),
        Err(ObliqueViewportError::EmptyFrame { dimensions: [0, 1] })
    );
    for geometry in [
        ([0.0, 0.0], [1.0, f64::INFINITY], [0.0, 0.0, 1.0, 1.0]),
        ([0.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
        ([0.0, 0.0], [1.0, 1.0], [0.0, 0.0, 0.0, 1.0]),
        ([f64::MAX, 0.0], [f64::MAX, 1.0], [0.0, 0.0, f64::MAX, 1.0]),
        (
            [9_007_199_254_740_992.0, 0.0],
            [1.0, 1.0],
            [9_007_199_254_740_992.0, 0.0, 9_007_199_254_740_996.0, 2.0],
        ),
        (
            [9_007_199_254_740_994.0, 0.0],
            [2.0, 1.0],
            [9_007_199_254_740_994.0, 0.0, 9_007_199_254_740_998.0, 1.0],
        ),
    ] {
        assert_eq!(
            ObliqueViewport::try_new(geometry.0, geometry.1, [2, 1], geometry.2),
            Err(ObliqueViewportError::InvalidImageGeometry)
        );
    }
    assert_eq!(
        ObliqueViewport::try_new(
            [9_007_199_254_740_994.0, 0.0],
            [2.0, 1.0],
            [4, 1],
            [9_007_199_254_740_994.0, 0.0, 9_007_199_254_741_002.0, 1.0],
        ),
        Err(ObliqueViewportError::InvalidImageGeometry)
    );
}

#[test]
fn visible_pixel_bounds_exclude_large_half_open_edges() {
    let origin = 18_014_398_509_481_984.0;
    let right_edge = 18_014_398_509_481_992.0;
    let viewport = ObliqueViewport::try_new(
        [origin, 0.0],
        [8.0, 1.0],
        [1, 1],
        [origin, 0.0, right_edge, 1.0],
    )
    .expect("representable frame geometry");
    let bounds = viewport
        .visible_pixel_bounds()
        .expect("one visible host-pixel centre");
    assert_eq!(bounds, [origin, right_edge - 4.0, 0.0, 0.0]);
    assert!(bounds[1] < right_edge);
    assert_eq!(
        viewport.map(ViewportPoint::new(bounds[1], 0.5)),
        Some([0.0, 0.0])
    );
}

#[test]
fn exact_centres_on_the_screen_precision_grid_remain_distinct() {
    for (origin, pixel_size, right_edge) in [
        (9_007_199_254_740_994.0, 3.0, 9_007_199_254_741_000.0),
        (9_007_199_254_740_991.0, 2.0, 9_007_199_254_740_996.0),
    ] {
        let viewport = ObliqueViewport::try_new(
            [origin, 0.0],
            [pixel_size, 1.0],
            [2, 1],
            [origin, 0.0, right_edge, 1.0],
        )
        .expect("distinct exact pixel centres at the binary64 spacing");

        for column in [0.0, 1.0] {
            let screen = viewport
                .screen_point([column, 0.0])
                .expect("pixel centre remains representable");
            assert_eq!(
                viewport.map(ViewportPoint::new(screen[0], screen[1])),
                Some([column, 0.0])
            );
        }
    }
}

#[test]
fn pixel_centres_map_without_overflowing_opposite_large_coordinates() {
    let pixel_size = 2.0_f64.powi(1022);
    let origin = -3.0 * pixel_size;
    let right_edge = 3.0 * pixel_size;
    let viewport = ObliqueViewport::try_new(
        [origin, 0.0],
        [pixel_size, 1.0],
        [6, 1],
        [origin, 0.0, right_edge, 1.0],
    )
    .expect("finite affine plane extent");
    let screen = viewport
        .screen_point([4.0, 0.0])
        .expect("finite pixel centre");
    assert_eq!(screen[0], 1.5 * pixel_size);
    assert_eq!(
        viewport.map(ViewportPoint::new(screen[0], screen[1])),
        Some([4.0, 0.0])
    );
}

#[test]
fn exact_centres_are_recovered_independently_for_each_axis() {
    let origin = 9_007_199_254_740_994.0;
    let viewport = ObliqueViewport::try_new(
        [origin, 0.0],
        [3.0, 1.0],
        [2, 2],
        [origin, 0.0, origin + 6.0, 2.0],
    )
    .expect("finite image rectangle over the host precision boundary");

    assert_eq!(
        viewport.map(ViewportPoint::new(origin + 4.0, 0.5)),
        Some([1.0, 0.0])
    );
    assert_eq!(
        viewport.map(ViewportPoint::new(origin + 4.0, 0.75)),
        Some([1.0, 0.25])
    );
    assert_eq!(
        viewport.map(ViewportPoint::new(origin, 1.5)),
        Some([0.0, 1.0])
    );
}

#[test]
fn viewport_mapping_preserves_rotated_anisotropic_patient_coordinates() {
    let volume = rotated_anisotropic_volume();
    let plane = ReslicePlane::try_new(
        &volume,
        [7.0, 23.0, 30.75],
        [0.0, 1.0, 0.25],
        [-3.0, 0.0, 0.0],
        [0.0; 3],
        [4, 3],
        1,
        ResliceInterpolation::Linear,
    )
    .expect("oblique plane lies inside the rotated anisotropic volume");
    let viewport = ObliqueViewport::try_new(
        [200.0, 100.0],
        [1.25, 0.75],
        [4, 3],
        [190.0, 90.0, 230.0, 130.0],
    )
    .expect("rendered plane geometry");
    viewport
        .validate_dimensions(plane.dimensions())
        .expect("rendered and physical plane dimensions agree");
    assert_eq!(
        viewport.validate_dimensions([5, 4]),
        Err(ObliqueViewportError::FrameDimensionsMismatch {
            viewport: [4, 3],
            plane: [5, 4],
        })
    );

    let screen = viewport
        .screen_point([1.0, 1.0])
        .expect("central reslice pixel has a screen centre");
    let pixel = viewport
        .map(ViewportPoint::new(screen[0], screen[1]))
        .expect("screen centre maps into the rendered plane");
    let patient = plane
        .patient_at_pixel(pixel)
        .expect("reslice pixel maps to patient space");
    assert_eq!(
        patient,
        PatientPointMm::try_new([4.0, 24.0, 31.0]).expect("finite patient point")
    );
}

fn rotated_anisotropic_volume() -> LoadedVolume {
    let mut volume = crate::app::tests::test_volume([5, 5, 5]);
    volume.spacing = [2.0, 3.0, 0.5];
    volume.origin = [10.0, 20.0, 30.0];
    volume.direction = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    volume
}
