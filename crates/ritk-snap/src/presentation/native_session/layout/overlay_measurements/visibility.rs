use super::{Annotation, AnnotationLocation};

pub(super) fn annotation_visible(annotation: &Annotation, plane: usize, slice: usize) -> bool {
    if let Some(AnnotationLocation {
        plane: recorded,
        slice: index,
    }) = annotation.location()
    {
        return recorded == plane && index == slice;
    }
    matches!(
        annotation,
        Annotation::Length { plane: recorded, .. }
            | Annotation::PatientLength { plane: recorded, .. } if *recorded == plane
    )
}
