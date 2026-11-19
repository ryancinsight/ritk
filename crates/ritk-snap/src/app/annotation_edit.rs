use super::pointer_bindings::{AnnotationMoveKind, PendingAnnotationMove};
use super::state::SnapApp;
use super::viewer_viewport::ViewerViewport;
use crate::presentation::ViewportPoint;
use crate::tools::interaction::ImagePoint;

mod polygon;

impl SnapApp {
    pub(crate) fn begin_annotation_move(
        &mut self,
        viewport: ViewerViewport,
        screen: ViewportPoint,
        point: ImagePoint,
    ) -> bool {
        let Some(annotation_index) = self.pointer.selected_annotation else {
            return false;
        };
        let location = self.annotation_location();
        let Some(original) = self.annotations.get(annotation_index) else {
            return false;
        };
        if original.location() != Some(location) {
            return false;
        }
        let (kind, candidate) = if let Some(node_index) =
            polygon::nearest_node(original, viewport, screen)
        {
            (
                AnnotationMoveKind::Polygon {
                    node_index,
                    inserted: false,
                },
                original.clone(),
            )
        } else {
            let Ok(spacing) = self.slice_plane_spacing() else {
                return false;
            };
            let Some((edge, fraction)) = polygon::nearest_edge(original, viewport, screen) else {
                return false;
            };
            let Some(candidate) = polygon::insert_node(original, edge, fraction, spacing) else {
                return false;
            };
            (
                AnnotationMoveKind::Polygon {
                    node_index: edge + 1,
                    inserted: true,
                },
                candidate,
            )
        };
        self.pointer.pending_annotation_move = Some(PendingAnnotationMove {
            annotation_index,
            source_location: location,
            original: original.clone(),
            kind,
            start: point,
            current: point,
            candidate,
        });
        true
    }

    pub(crate) fn preview_annotation_move(&mut self, point: ImagePoint) -> bool {
        let location = self.annotation_location();
        let Ok(spacing) = self.slice_plane_spacing() else {
            return false;
        };
        let Some(pending) = self.pointer.pending_annotation_move.as_mut() else {
            return false;
        };
        if pending.source_location != location {
            self.on_drag_end(None);
            return false;
        }
        let AnnotationMoveKind::Polygon {
            node_index,
            inserted,
        } = pending.kind;
        let source = if inserted {
            &pending.candidate
        } else {
            &pending.original
        };
        let Some(candidate) = polygon::move_node(source, node_index, point, spacing) else {
            return false;
        };
        pending.current = point;
        pending.candidate = candidate;
        true
    }

    pub(super) fn commit_annotation_move(&mut self) {
        let Some(pending) = self.pointer.pending_annotation_move.take() else {
            return;
        };
        if pending.source_location == self.annotation_location()
            && (matches!(
                pending.kind,
                AnnotationMoveKind::Polygon { inserted: true, .. }
            ) || pending.start != pending.current)
            && let Some(annotation) = self.annotations.get_mut(pending.annotation_index)
            && *annotation == pending.original
        {
            *annotation = pending.candidate;
            self.bump_visual_revision();
        }
    }
}
