//! Geometry and value updates for editable polygon annotations.

use crate::tools::interaction::{Annotation, ImagePoint};
use metis_ui_lang::{nearest_polyline_segment, nearest_polyline_vertex, PathTopology};

const NODE_PICK_RADIUS_PIXELS: f64 = 6.0;

fn points(annotation: &Annotation) -> Option<(&[[f32; 2]], PathTopology)> {
    match annotation {
        Annotation::ClosedPolygon { points, .. } => Some((points, PathTopology::Closed)),
        Annotation::OpenPolygon { points, .. } => Some((points, PathTopology::Open)),
        _ => None,
    }
}

fn projected(point: [f32; 2], pixel_size: [f64; 2]) -> [f64; 2] {
    [
        f64::from(point[1]) * pixel_size[0],
        f64::from(point[0]) * pixel_size[1],
    ]
}

pub(crate) fn nearest_node(
    annotation: &Annotation,
    point: ImagePoint,
    pixel_size: [f64; 2],
) -> Option<usize> {
    let (points, _) = points(annotation)?;
    let target = projected([point.y(), point.x()], pixel_size);
    let hit = nearest_polyline_vertex(
        points
            .iter()
            .copied()
            .map(|value| projected(value, pixel_size)),
        target,
    )
    .ok()?;
    if hit.distance() > NODE_PICK_RADIUS_PIXELS {
        return None;
    }
    Some(hit.index())
}

pub(crate) fn nearest_segment(
    annotation: &Annotation,
    point: ImagePoint,
    pixel_size: [f64; 2],
) -> Option<usize> {
    let (points, topology) = points(annotation)?;
    let target = projected([point.y(), point.x()], pixel_size);
    let hit = nearest_polyline_segment(
        points
            .iter()
            .copied()
            .map(|value| projected(value, pixel_size)),
        target,
        topology,
    )
    .ok()?;
    if hit.distance() > NODE_PICK_RADIUS_PIXELS {
        return None;
    }
    Some(hit.start_index())
}

pub(crate) fn move_node(
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

pub(crate) fn insert_node(
    original: &Annotation,
    segment_index: usize,
    point: ImagePoint,
    spacing: [f64; 2],
) -> Option<Annotation> {
    let mut candidate = original.clone();
    let point = [point.y(), point.x()];
    match &mut candidate {
        Annotation::ClosedPolygon { points, .. } | Annotation::OpenPolygon { points, .. } => {
            let open = matches!(original, Annotation::OpenPolygon { .. });
            if points.len() >= metis_platform::rasterizer::MAX_STROKE_POINTS
                || segment_index >= points.len()
                || (open && segment_index + 1 >= points.len())
            {
                return None;
            }
            points.insert(segment_index + 1, point);
        }
        _ => return None,
    }
    recompute(candidate, spacing)
}

fn recompute(mut candidate: Annotation, spacing: [f64; 2]) -> Option<Annotation> {
    let spacing = Annotation::validate_spacing(spacing).ok()?;
    let valid = match &mut candidate {
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
            *perimeter_mm = closed_perimeter(points, spacing);
            area_mm2.is_finite() && *area_mm2 > 0.0 && perimeter_mm.is_finite()
        }
        Annotation::OpenPolygon {
            points, length_mm, ..
        } => {
            if points.len() < 2 || !points.iter().flatten().all(|value| value.is_finite()) {
                return None;
            }
            *length_mm = Annotation::compute_polyline_length(points, spacing);
            length_mm.is_finite() && *length_mm > 0.0
        }
        _ => false,
    };
    valid.then_some(candidate)
}

fn closed_perimeter(points: &[[f32; 2]], spacing: [f32; 2]) -> f32 {
    let Some((&first, _)) = points.split_first() else {
        return 0.0;
    };
    let Some(&last) = points.last() else {
        return 0.0;
    };
    Annotation::compute_polyline_length(points, spacing)
        + Annotation::compute_polyline_length(&[last, first], spacing)
}

#[cfg(test)]
mod tests {
    use super::{insert_node, move_node, nearest_node, nearest_segment};
    use crate::tools::interaction::{Annotation, AnnotationLocation, ImagePoint};

    fn open() -> Annotation {
        Annotation::OpenPolygon {
            location: Some(AnnotationLocation { plane: 0, slice: 2 }),
            points: vec![[2.0, 0.0], [4.0, 8.0], [8.0, 10.0]],
            length_mm: 14.0,
        }
    }

    #[test]
    fn node_move_recomputes_length_and_preserves_location() {
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
    fn invalid_node_move_rolls_back_as_no_candidate() {
        assert!(move_node(&open(), 1, ImagePoint::new(f32::NAN, 2.0), [1.0, 1.0]).is_none());
    }

    #[test]
    fn segment_insertion_uses_metis_path_index() {
        let annotation = open();
        assert_eq!(
            nearest_node(&annotation, ImagePoint::new(0.0, 2.0), [1.0, 1.0]),
            Some(0)
        );
        assert_eq!(
            nearest_segment(&annotation, ImagePoint::new(2.0, 6.0), [1.0, 1.0]),
            Some(0)
        );
        let inserted = insert_node(&annotation, 0, ImagePoint::new(6.0, 2.0), [1.0, 1.0])
            .expect("invariant: insertion on an open segment preserves geometry");
        assert!(
            matches!(inserted, Annotation::OpenPolygon { points, .. } if points[1] == [2.0, 6.0])
        );
    }
}
