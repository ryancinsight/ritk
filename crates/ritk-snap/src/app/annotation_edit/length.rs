use super::super::pointer_bindings::LengthMove;
use super::super::polygon_selection::PICK_RADIUS_PIXELS;
use super::super::viewer_viewport::ViewerViewport;
use crate::presentation::ViewportPoint;
use crate::tools::interaction::{Annotation, ImagePoint};
use metis_ui_lang::nearest_polyline_vertex;

pub(super) fn pick(
    annotation: &Annotation,
    viewport: ViewerViewport,
    target: ViewportPoint,
) -> Option<LengthMove> {
    let Annotation::Length { p1, p2, .. } = annotation else {
        return None;
    };
    let projected = [*p1, *p2].map(|point| viewport.project(point));
    let target = [target.x(), target.y()];
    let vertex = nearest_polyline_vertex(projected, target).ok()?;
    if vertex.distance() <= PICK_RADIUS_PIXELS {
        return match vertex.index() {
            0 => Some(LengthMove::FirstEndpoint),
            1 => Some(LengthMove::SecondEndpoint),
            _ => None,
        };
    }
    None
}

pub(super) fn move_candidate(
    original: &Annotation,
    mode: LengthMove,
    current: ImagePoint,
    spacing: [f64; 2],
) -> Option<Annotation> {
    let Annotation::Length { p1, p2, .. } = original else {
        return None;
    };
    let mut candidate = original.clone();
    let (next_p1, next_p2) = match mode {
        LengthMove::FirstEndpoint => ([current.y(), current.x()], *p2),
        LengthMove::SecondEndpoint => (*p1, [current.y(), current.x()]),
    };
    let Annotation::Length {
        p1, p2, length_mm, ..
    } = &mut candidate
    else {
        return None;
    };
    *p1 = next_p1;
    *p2 = next_p2;
    *length_mm = Annotation::compute_length_checked(*p1, *p2, spacing).ok()?;
    Some(candidate)
}
