//! Measurement line work projected over native image panes.
//!
//! Completed annotations and the gesture in progress are stored in continuous
//! source-image coordinates. They pass through the same projection as the
//! pixels, so a measurement stays on its anatomy under zoom, pan and rotation,
//! and each segment is clipped to the visible image so it cannot paint chrome.

use anyhow::Result;
use metis_platform::Color;
use metis_ui_lang::DisplayList;

use super::super::frame::RenderedView;
use super::geometry::{
    screen_coordinate, source_point_to_screen, visible_image_bounds, NativeViewport,
};
use crate::tools::interaction::{Annotation, PlacedAnnotation};

pub(crate) const MEASUREMENT_COLOR: Color = Color::rgba(255, 255, 0, 255);

/// Build the line work for one pane.
///
/// # Errors
/// Returns an error when a projected coordinate is outside the host range or
/// the display list cannot grow.
pub(crate) fn measurement_overlay(
    view: &RenderedView,
    viewport: &NativeViewport,
    annotations: &[PlacedAnnotation],
) -> Result<DisplayList> {
    let mut list = DisplayList::default();
    let Some(clip) = visible_image_bounds(viewport) else {
        return Ok(list);
    };
    let pane = Pane {
        view,
        viewport,
        clip,
    };
    for placed in annotations
        .iter()
        .filter(|placed| placed.is_visible_on(view.axis, view.slice_index))
    {
        pane.draw_annotation(&placed.annotation, &mut list)?;
    }
    Ok(list)
}

struct Pane<'a> {
    view: &'a RenderedView,
    viewport: &'a NativeViewport,
    clip: [f64; 4],
}

impl Pane<'_> {
    /// Project a stored `[row, col]` point to device coordinates.
    fn point(&self, image: [f32; 2]) -> Result<Option<[f64; 2]>> {
        source_point_to_screen(
            self.view,
            self.viewport,
            [f64::from(image[1]), f64::from(image[0])],
        )
    }

    fn segment(
        &self,
        list: &mut DisplayList,
        a: [f64; 2],
        b: [f64; 2],
        color: Color,
    ) -> Result<()> {
        if let Some((start, end)) = clip_segment(a, b, self.clip) {
            list.append_line(screen_point(start)?, screen_point(end)?, color)?;
        }
        Ok(())
    }

    fn image_segment(
        &self,
        list: &mut DisplayList,
        a: [f32; 2],
        b: [f32; 2],
        color: Color,
    ) -> Result<()> {
        if let (Some(a), Some(b)) = (self.point(a)?, self.point(b)?) {
            self.segment(list, a, b, color)?;
        }
        Ok(())
    }

    fn draw_annotation(&self, annotation: &Annotation, list: &mut DisplayList) -> Result<()> {
        match annotation {
            Annotation::Length { p1, p2, .. } => {
                self.image_segment(list, *p1, *p2, MEASUREMENT_COLOR)
            }
            Annotation::Angle { p1, p2, p3, .. } => {
                self.image_segment(list, *p1, *p2, MEASUREMENT_COLOR)?;
                self.image_segment(list, *p2, *p3, MEASUREMENT_COLOR)
            }
            Annotation::PatientLength(_)
            | Annotation::RoiRect { .. }
            | Annotation::RoiEllipse { .. }
            | Annotation::HuPoint { .. } => Ok(()),
        }
    }
}

fn screen_point([x, y]: [f64; 2]) -> Result<(i32, i32)> {
    Ok((
        screen_coordinate(x, "measurement x")?,
        screen_coordinate(y, "measurement y")?,
    ))
}

/// Clip a segment to `[left, top, right, bottom]` (Liang and Barsky, 1984).
fn clip_segment(a: [f64; 2], b: [f64; 2], clip: [f64; 4]) -> Option<([f64; 2], [f64; 2])> {
    let [left, top, right, bottom] = clip;
    let delta = [b[0] - a[0], b[1] - a[1]];
    let (mut enter, mut leave) = (0.0_f64, 1.0_f64);
    for (direction, distance) in [
        (-delta[0], a[0] - left),
        (delta[0], right - a[0]),
        (-delta[1], a[1] - top),
        (delta[1], bottom - a[1]),
    ] {
        if direction == 0.0 {
            if distance < 0.0 {
                return None;
            }
        } else {
            let crossing = distance / direction;
            if direction < 0.0 {
                enter = enter.max(crossing);
            } else {
                leave = leave.min(crossing);
            }
            if enter > leave {
                return None;
            }
        }
    }
    let at = |t: f64| [a[0] + t * delta[0], a[1] + t * delta[1]];
    Some((at(enter), at(leave)))
}

#[cfg(test)]
mod tests {
    use super::clip_segment;

    const CLIP: [f64; 4] = [0.0, 0.0, 10.0, 10.0];

    #[test]
    fn a_segment_inside_the_clip_is_unchanged() {
        assert_eq!(
            clip_segment([1.0, 2.0], [3.0, 4.0], CLIP),
            Some(([1.0, 2.0], [3.0, 4.0]))
        );
    }

    #[test]
    fn a_crossing_segment_is_cut_at_the_boundary() {
        assert_eq!(
            clip_segment([-5.0, 5.0], [15.0, 5.0], CLIP),
            Some(([0.0, 5.0], [10.0, 5.0]))
        );
    }

    #[test]
    fn a_segment_outside_the_clip_is_rejected() {
        assert_eq!(clip_segment([11.0, 0.0], [20.0, 9.0], CLIP), None);
    }

    #[test]
    fn a_parallel_segment_outside_the_clip_is_rejected() {
        assert_eq!(clip_segment([-1.0, 2.0], [-1.0, 8.0], CLIP), None);
    }
}
