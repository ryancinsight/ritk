use super::{annotation_visible, Annotation, AnnotationLocation};

#[test]
fn anchored_annotation_requires_matching_plane_and_slice() {
    let annotation = Annotation::HuPoint {
        location: Some(AnnotationLocation { plane: 1, slice: 4 }),
        pos: [0.0, 0.0],
        value: -10.0,
    };

    assert!(annotation_visible(&annotation, 1, 4));
    assert!(!annotation_visible(&annotation, 0, 4));
    assert!(!annotation_visible(&annotation, 1, 3));
}

#[test]
fn legacy_point_without_image_location_does_not_claim_an_image() {
    let annotation = Annotation::HuPoint {
        location: None,
        pos: [0.0, 0.0],
        value: -10.0,
    };
    assert!(!annotation_visible(&annotation, 0, 0));
    assert!(!annotation_visible(&annotation, 2, 17));
}
