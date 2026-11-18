use super::super::polygon_selection::{polygon_path, PICK_RADIUS_PIXELS};
use super::super::viewer_viewport::ViewerViewport;
use crate::presentation::ViewportPoint;
use crate::tools::interaction::{Annotation, ImagePoint};
use eunomia::FloatElement;
use metis_ui_lang::{nearest_polyline_segment, nearest_polyline_vertex, PathTopology};

pub(super) fn nearest_node(
    annotation: &Annotation,
    viewport: ViewerViewport,
    target: ViewportPoint,
) -> Option<usize> {
    let (points, _) = polygon_path(annotation)?;
    let hit = nearest_polyline_vertex(
        points.iter().copied().map(|point| viewport.project(point)),
        [target.x(), target.y()],
    )
    .ok()?;
    (hit.distance() <= PICK_RADIUS_PIXELS).then_some(hit.index())
}

pub(super) fn move_node(
    original: &Annotation,
    node_index: usize,
    point: ImagePoint,
    spacing: [f64; 2],
) -> Option<Annotation> {
    let mut candidate = original.clone();
    let point = [point.y(), point.x()];
    match &mut candidate {
        Annotation::ClosedPolygon { points, .. } | Annotation::OpenPolygon { points, .. } => {
            *points.get_mut(node_index)? = point;
        }
        _ => return None,
    }
    recompute(candidate, spacing)
}

pub(super) fn nearest_edge(
    annotation: &Annotation,
    viewport: ViewerViewport,
    target: ViewportPoint,
) -> Option<(usize, f64)> {
    let (points, topology) = super::super::polygon_selection::polygon_path(annotation)?;
    let hit = nearest_polyline_segment(
        points.iter().copied().map(|point| viewport.project(point)),
        [target.x(), target.y()],
        topology,
    )
    .ok()?;
    (hit.distance() <= super::super::polygon_selection::PICK_RADIUS_PIXELS
        && hit.fraction() > 0.0
        && hit.fraction() < 1.0)
        .then_some((hit.start_index(), hit.fraction()))
}

pub(super) fn insert_node(
    original: &Annotation,
    edge: usize,
    fraction: f64,
    spacing: [f64; 2],
) -> Option<Annotation> {
    if !fraction.is_finite() || fraction <= 0.0 || fraction >= 1.0 {
        return None;
    }
    let mut candidate = original.clone();
    let (start, next) = {
        let (points, topology) = super::super::polygon_selection::polygon_path(&candidate)?;
        let next = match topology {
            PathTopology::Open => points.get(edge + 1).copied()?,
            PathTopology::Closed => points.get(edge + 1).copied().unwrap_or(points[0]),
        };
        (*points.get(edge)?, next)
    };
    let fraction = <f32 as FloatElement>::from_f64(fraction);
    let point = [
        start[0] + (next[0] - start[0]) * fraction,
        start[1] + (next[1] - start[1]) * fraction,
    ];
    match &mut candidate {
        Annotation::ClosedPolygon { points, .. } | Annotation::OpenPolygon { points, .. } => {
            points.insert(edge + 1, point);
        }
        _ => return None,
    }
    recompute(candidate, spacing)
}

fn recompute(mut candidate: Annotation, spacing: [f64; 2]) -> Option<Annotation> {
    let spacing = Annotation::validate_spacing(spacing).ok()?;
    match &mut candidate {
        Annotation::ClosedPolygon {
            points,
            area_mm2,
            perimeter_mm,
            ..
        } => {
            if points.len() < 3 || !points.iter().flatten().all(|value| value.is_finite()) {
                return None;
            }
            *area_mm2 = Annotation::compute_polygon_area(points, spacing);
            *perimeter_mm = Annotation::compute_polygon_perimeter(points, spacing);
            (area_mm2.is_finite() && *area_mm2 > 0.0 && perimeter_mm.is_finite())
                .then_some(candidate)
        }
        Annotation::OpenPolygon {
            points, length_mm, ..
        } => {
            if points.len() < 2 || !points.iter().flatten().all(|value| value.is_finite()) {
                return None;
            }
            *length_mm = Annotation::compute_polyline_length(points, spacing);
            (length_mm.is_finite() && *length_mm > 0.0).then_some(candidate)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{insert_node, move_node};
    use crate::tools::interaction::{Annotation, AnnotationLocation, ImagePoint};

    fn open() -> Annotation {
        Annotation::OpenPolygon {
            location: Some(AnnotationLocation { plane: 0, slice: 2 }),
            points: vec![[2.0, 0.0], [4.0, 8.0], [8.0, 10.0]],
            length_mm: 14.0,
        }
    }

    #[test]
    fn moving_a_node_recomputes_length_and_preserves_location() {
        let moved = move_node(&open(), 1, ImagePoint::new(10.0, 2.0), [1.0, 1.0])
            .expect("invariant: finite non-degenerate open path remains editable");
        match moved {
            Annotation::OpenPolygon {
                points,
                length_mm,
                location,
            } => {
                assert_eq!(points[1], [2.0, 10.0]);
                assert_eq!(length_mm, 16.0);
                assert_eq!(location, Some(AnnotationLocation { plane: 0, slice: 2 }));
            }
            _ => panic!("invariant: open polygon edit preserves annotation kind"),
        }
    }

    #[test]
    fn moving_a_closed_vertex_recomputes_area() {
        let moved = move_node(
            &Annotation::ClosedPolygon {
                location: Some(AnnotationLocation { plane: 0, slice: 2 }),
                points: vec![[0.0, 0.0], [0.0, 4.0], [3.0, 4.0], [3.0, 0.0]],
                area_mm2: 12.0,
                perimeter_mm: 14.0,
            },
            1,
            ImagePoint::new(2.0, 0.0),
            [1.0, 1.0],
        )
        .expect("invariant: finite closed polygon remains editable");
        assert!(matches!(
            moved,
            Annotation::ClosedPolygon { area_mm2, .. } if area_mm2 == 9.0
        ));
    }

    #[test]
    fn endpoint_insertion_is_rejected() {
        assert_eq!(insert_node(&open(), 0, 0.0, [1.0, 1.0]), None);
    }
}
