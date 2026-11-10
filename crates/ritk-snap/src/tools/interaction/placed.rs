//! Annotations bound to the image plane they were placed on.

use super::Annotation;

/// The image plane an annotation was placed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AnnotationPlane {
    /// Viewing axis index: 0 axial, 1 coronal, 2 sagittal.
    pub axis: usize,
    /// Slice index along that axis.
    pub slice: usize,
}

/// A completed annotation and the plane it belongs to.
///
/// The annotation's fields stay at the top level of its serialized form, so a
/// session written before planes were recorded still loads. Such an annotation
/// has no plane and is shown on every slice, as it was when it was saved.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PlacedAnnotation {
    /// Plane the annotation was placed on, when it was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plane: Option<AnnotationPlane>,
    /// The measurement itself.
    #[serde(flatten)]
    pub annotation: Annotation,
}

impl PlacedAnnotation {
    /// Bind `annotation` to `plane`.
    #[must_use]
    pub const fn on_plane(plane: AnnotationPlane, annotation: Annotation) -> Self {
        Self {
            plane: Some(plane),
            annotation,
        }
    }

    /// Whether the annotation is shown on `slice` along `axis`.
    #[must_use]
    pub fn is_visible_on(&self, axis: usize, slice: usize) -> bool {
        self.plane
            .is_none_or(|plane| plane.axis == axis && plane.slice == slice)
    }
}

impl From<Annotation> for PlacedAnnotation {
    fn from(annotation: Annotation) -> Self {
        Self {
            plane: None,
            annotation,
        }
    }
}
