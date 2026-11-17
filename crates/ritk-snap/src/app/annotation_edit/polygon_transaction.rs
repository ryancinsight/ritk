//! Transactional polygon pointer edits.

use super::super::pointer_bindings::PendingPolygonEdit;
use super::super::state::SnapApp;
use super::polygon;
use crate::tools::interaction::{AnnotationLocation, ImagePoint};

impl SnapApp {
    pub(crate) fn begin_polygon_edit(&mut self, point: ImagePoint, pixel_size: [f64; 2]) -> bool {
        let Some(annotation_index) = self.pointer.selected_annotation else {
            return false;
        };
        let location = self.annotation_location();
        let Some(original) = self.annotations.get(annotation_index) else {
            return false;
        };
        if !original.visible_on_image(location) {
            return false;
        }
        let Ok(spacing) = self.slice_plane_spacing() else {
            return false;
        };
        let pending = if let Some(node_index) = polygon::nearest_node(original, point, pixel_size) {
            Some(PendingPolygonEdit::Move {
                annotation_index,
                source_location: location,
                original: original.clone(),
                node_index,
                start: point,
                current: point,
                candidate: original.clone(),
            })
        } else if let Some(segment_index) = polygon::nearest_segment(original, point, pixel_size) {
            polygon::insert_node(original, segment_index, point, spacing).map(|candidate| {
                PendingPolygonEdit::Insert {
                    annotation_index,
                    source_location: location,
                    original: original.clone(),
                    segment_index,
                    start: point,
                    current: point,
                    candidate,
                }
            })
        } else {
            None
        };
        let Some(pending) = pending else {
            return false;
        };
        self.pointer.pending_polygon_edit = Some(pending);
        true
    }

    #[cfg(windows)]
    pub(crate) fn polygon_edit_preview(
        &self,
        index: usize,
        location: AnnotationLocation,
    ) -> Option<&crate::tools::interaction::Annotation> {
        match self.pointer.pending_polygon_edit.as_ref()? {
            PendingPolygonEdit::Move {
                annotation_index,
                source_location,
                candidate,
                ..
            }
            | PendingPolygonEdit::Insert {
                annotation_index,
                source_location,
                candidate,
                ..
            } if *annotation_index == index && *source_location == location => Some(candidate),
            _ => None,
        }
    }

    pub(crate) fn preview_polygon_edit(&mut self, point: ImagePoint) -> bool {
        let Some(pending) = self.pointer.pending_polygon_edit.take() else {
            return false;
        };
        if pending_source_location(&pending) != self.annotation_location() {
            return false;
        }
        let Ok(spacing) = self.slice_plane_spacing() else {
            return false;
        };
        let updated = match pending {
            PendingPolygonEdit::Move {
                annotation_index,
                source_location,
                original,
                node_index,
                start,
                ..
            } => polygon::move_node(&original, node_index, point, spacing).map(|candidate| {
                PendingPolygonEdit::Move {
                    annotation_index,
                    source_location,
                    original,
                    node_index,
                    start,
                    current: point,
                    candidate,
                }
            }),
            PendingPolygonEdit::Insert {
                annotation_index,
                source_location,
                original,
                segment_index,
                start,
                ..
            } => polygon::insert_node(&original, segment_index, point, spacing).map(|candidate| {
                PendingPolygonEdit::Insert {
                    annotation_index,
                    source_location,
                    original,
                    segment_index,
                    start,
                    current: point,
                    candidate,
                }
            }),
        };
        let Some(updated) = updated else {
            return false;
        };
        self.pointer.pending_polygon_edit = Some(updated);
        true
    }

    pub(crate) fn commit_polygon_edit(&mut self) -> bool {
        let Some(pending) = self.pointer.pending_polygon_edit.take() else {
            return false;
        };
        if pending_source_location(&pending) != self.annotation_location() {
            return false;
        }
        let (index, original, candidate, changed) = match pending {
            PendingPolygonEdit::Move {
                annotation_index,
                start,
                current,
                original,
                candidate,
                ..
            } => (annotation_index, original, candidate, start != current),
            PendingPolygonEdit::Insert {
                annotation_index,
                original,
                candidate,
                ..
            } => (annotation_index, original, candidate, true),
        };
        if !changed {
            return false;
        }
        let Some(annotation) = self.annotations.get_mut(index) else {
            return false;
        };
        if *annotation != original {
            return false;
        }
        *annotation = candidate;
        true
    }

    pub(crate) fn abandon_polygon_edit_if_source_changed(&mut self) -> bool {
        let changed = self
            .pointer
            .pending_polygon_edit
            .as_ref()
            .is_some_and(|pending| pending_source_location(pending) != self.annotation_location());
        if changed {
            self.pointer.pending_polygon_edit = None;
            self.on_drag_end(None);
        }
        changed
    }
}

fn pending_source_location(pending: &PendingPolygonEdit) -> AnnotationLocation {
    match pending {
        PendingPolygonEdit::Move {
            source_location, ..
        }
        | PendingPolygonEdit::Insert {
            source_location, ..
        } => *source_location,
    }
}
