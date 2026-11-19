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
    match vertex.index() {
        0 if vertex.distance() <= PICK_RADIUS_PIXELS => Some(LengthMove::FirstEndpoint),
        1 if vertex.distance() <= PICK_RADIUS_PIXELS => Some(LengthMove::SecondEndpoint),
        _ => None,
    }
}

pub(super) fn move_candidate(
    original: &Annotation,
    mode: LengthMove,
    current: ImagePoint,
    spacing: [f64; 2],
) -> Option<Annotation> {
    let mut candidate = original.clone();
    let Annotation::Length {
        p1, p2, length_mm, ..
    } = &mut candidate
    else {
        return None;
    };
    match mode {
        LengthMove::FirstEndpoint => *p1 = [current.y(), current.x()],
        LengthMove::SecondEndpoint => *p2 = [current.y(), current.x()],
    }
    *length_mm = Annotation::compute_length_checked(*p1, *p2, spacing).ok()?;
    Some(candidate)
}
