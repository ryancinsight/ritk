//! Event-time content coordinates expressed as image-relative fractions.

/// Normalizes a local position by the content dimensions measured with it.
///
/// Negative and beyond-content positions remain outside the content box so
/// border, padding and captured-pointer events cannot be clamped into the image.
pub(crate) fn content_fraction(point: [f64; 2], event_size: [f64; 2]) -> Option<[f64; 2]> {
    if !point.iter().all(|value| value.is_finite())
        || !event_size
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    {
        return None;
    }
    let mapped = [point[0] / event_size[0], point[1] / event_size[1]];
    mapped
        .iter()
        .all(|value| value.is_finite())
        .then_some(mapped)
}

#[cfg(test)]
mod tests {
    use super::content_fraction;
    use crate::presentation::{
        PointerButton, PointerGesture, PresentationDispatcher, PresentationEvent, ViewerAction,
        ViewportPoint,
    };

    #[test]
    fn resize_between_event_and_drain_preserves_image_fraction() {
        assert_eq!(
            content_fraction([20.125, 30.1875], [80.5, 40.25]),
            Some([0.25, 0.75])
        );
        assert_eq!(
            content_fraction([40.25, 90.5625], [161.0, 120.75]),
            Some([0.25, 0.75])
        );
    }

    #[test]
    fn same_image_point_remains_a_click_across_resize_batches() {
        let mut dispatcher = PresentationDispatcher::new();
        let [x, y] = content_fraction([20.125, 30.1875], [80.5, 40.25])
            .expect("finite measured content point");
        let pressed = dispatcher
            .dispatch(&[PresentationEvent::PointerDown {
                x,
                y,
                button: PointerButton::Left,
            }])
            .expect("valid pointer press");
        assert_eq!(
            pressed.as_ref(),
            &[ViewerAction::PointerPressed {
                button: PointerButton::Left,
                position: ViewportPoint::new(0.25, 0.75),
            }]
        );
        let [x, y] = content_fraction([40.25, 90.5625], [161.0, 120.75])
            .expect("finite resized content point");
        let released = dispatcher
            .dispatch(&[PresentationEvent::PointerUp {
                x,
                y,
                button: PointerButton::Left,
            }])
            .expect("valid pointer release");
        assert_eq!(
            released.as_ref(),
            &[ViewerAction::PointerReleased {
                button: PointerButton::Left,
                position: ViewportPoint::new(0.25, 0.75),
                gesture: PointerGesture::Click,
            }]
        );
    }

    #[test]
    fn padding_and_captured_positions_stay_outside() {
        for (point, expected) in [
            ([-1.0, 10.0], [-0.125, 2.5]),
            ([10.0, -1.0], [1.25, -0.25]),
            ([9.0, 5.0], [1.125, 1.25]),
        ] {
            assert_eq!(content_fraction(point, [8.0, 4.0]), Some(expected));
        }
    }

    #[test]
    fn invalid_measurements_and_unrepresentable_points_are_rejected() {
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(content_fraction([1.0; 2], [invalid, 1.0]), None);
            assert_eq!(content_fraction([1.0; 2], [1.0, invalid]), None);
        }
        assert_eq!(content_fraction([f64::NAN, 1.0], [1.0; 2]), None);
        assert_eq!(content_fraction([f64::MAX, 1.0], [0.5, 1.0]), None);
    }
}
