use super::*;

#[test]
fn test_annotation_state_empty() {
    let state = AnnotationState::new();
    assert_eq!(state.total_count(), 0);
    assert!(state.points.is_empty());
    assert!(state.contours.is_empty());
    assert!(state.polylines.is_empty());
}

#[test]
fn test_add_point() {
    let mut state = AnnotationState::new();
    let ann = PointAnnotation::with_label([1.0, 2.0, 3.0], 5);
    state.add_point(ann);
    assert_eq!(state.points.len(), 1);
    assert_eq!(state.points[0].position, Point::new([1.0, 2.0, 3.0]));
    assert_eq!(state.points[0].label_id, Some(LabelId(5)));
}

#[test]
fn test_add_contour_valid() {
    let mut state = AnnotationState::new();
    let pts = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.5, 1.0, 0.0]),
    ];
    state.add_contour(pts.clone()).expect("infallible: validated precondition");
    assert_eq!(state.contours.len(), 1);
    assert_eq!(state.contours[0], pts);
}

#[test]
fn test_add_contour_too_short() {
    let mut state = AnnotationState::new();
    let result = state.add_contour(vec![Point::new([0.0, 0.0, 0.0])]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            AnnotationError::TooFewPoints {
                kind: "contour",
                count: 1
            }
        ),
        "unexpected error variant: {:?}",
        err
    );
}

#[test]
fn test_add_polyline_valid() {
    let mut state = AnnotationState::new();
    let pts = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([2.0, 0.0, 0.0]),
        Point::new([3.0, 0.0, 0.0]),
    ];
    state.add_polyline(pts.clone()).expect("infallible: validated precondition");
    assert_eq!(state.polylines.len(), 1);
    assert_eq!(state.polylines[0], pts);
}

#[test]
fn test_add_polyline_too_short() {
    let mut state = AnnotationState::new();
    let result = state.add_polyline(vec![Point::new([0.0, 0.0, 0.0])]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            AnnotationError::TooFewPoints {
                kind: "polyline",
                count: 1
            }
        ),
        "unexpected error variant: {:?}",
        err
    );
}

#[test]
fn test_clear() {
    let mut state = AnnotationState::new();
    state.add_point(PointAnnotation::new([0.0, 0.0, 0.0]));
    state
        .add_contour(vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
        ])
        .expect("infallible: validated precondition");
    state
        .add_polyline(vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
        ])
        .expect("infallible: validated precondition");
    state.clear();
    assert_eq!(state.total_count(), 0);
    assert!(state.points.is_empty());
    assert!(state.contours.is_empty());
    assert!(state.polylines.is_empty());
}

#[test]
fn test_seeds_for_label() {
    let mut state = AnnotationState::new();
    state.add_point(PointAnnotation::with_label([1.0, 0.0, 0.0], 1));
    state.add_point(PointAnnotation::with_label([2.0, 0.0, 0.0], 1));
    state.add_point(PointAnnotation::with_label([3.0, 0.0, 0.0], 2));
    let seeds1 = state.seeds_for_label(1);
    assert_eq!(seeds1.len(), 2);
    assert_eq!(seeds1[0], Point::new([1.0, 0.0, 0.0]));
    assert_eq!(seeds1[1], Point::new([2.0, 0.0, 0.0]));
    let seeds2 = state.seeds_for_label(2);
    assert_eq!(seeds2.len(), 1);
    assert_eq!(seeds2[0], Point::new([3.0, 0.0, 0.0]));
}

#[test]
fn test_json_roundtrip() {
    let mut state = AnnotationState::new();
    state.add_point(PointAnnotation::with_label([10.0, 20.0, 30.0], 3));
    state
        .add_contour(vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.5, 1.0, 0.0]),
        ])
        .expect("infallible: validated precondition");
    state
        .add_polyline(vec![
            Point::new([5.0, 5.0, 5.0]),
            Point::new([6.0, 5.0, 5.0]),
        ])
        .expect("infallible: validated precondition");
    let json = state.to_json().expect("serialization must succeed");
    let restored = AnnotationState::from_json(&json).expect("deserialization must succeed");
    assert_eq!(restored.points.len(), state.points.len());
    assert_eq!(restored.points[0].position, Point::new([10.0, 20.0, 30.0]));
    assert_eq!(restored.points[0].label_id, Some(LabelId(3)));
    assert_eq!(restored.contours.len(), 1);
    assert_eq!(restored.contours[0].len(), 3);
    assert_eq!(restored.polylines.len(), 1);
    assert_eq!(restored.polylines[0].len(), 2);
}
