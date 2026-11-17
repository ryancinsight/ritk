use super::pointer_bindings::PendingPolygonMove;
use super::state::SnapApp;
use super::viewer_viewport::ViewerViewport;
use crate::presentation::ViewportPoint;
use crate::tools::interaction::ImagePoint;

mod polygon;

impl SnapApp {
    pub(crate) fn begin_polygon_vertex_move(
        &mut self,
        viewport: ViewerViewport,
        screen: ViewportPoint,
        point: ImagePoint,
    ) -> bool {
        let Some(annotation_index) = self.pointer.selected_polygon else {
            return false;
        };
        let location = self.annotation_location();
        let Some(original) = self.annotations.get(annotation_index) else {
            return false;
        };
        if original.location() != Some(location) {
            return false;
        }
        let Some(node_index) = polygon::nearest_node(original, viewport, screen) else {
            return false;
        };
        self.pointer.pending_polygon_move = Some(PendingPolygonMove {
            annotation_index,
            source_location: location,
            original: original.clone(),
            node_index,
            start: point,
            current: point,
            candidate: original.clone(),
        });
        true
    }

    pub(crate) fn preview_polygon_vertex_move(&mut self, point: ImagePoint) -> bool {
        let Some(pending) = self.pointer.pending_polygon_move.take() else {
            return false;
        };
        if pending.source_location != self.annotation_location() {
            return false;
        }
        let Ok(spacing) = self.slice_plane_spacing() else {
            return false;
        };
        let Some(candidate) =
            polygon::move_node(&pending.original, pending.node_index, point, spacing)
        else {
            return false;
        };
        self.pointer.pending_polygon_move = Some(PendingPolygonMove {
            current: point,
            candidate,
            ..pending
        });
        true
    }

    pub(crate) fn commit_polygon_vertex_move(&mut self) -> bool {
        let Some(pending) = self.pointer.pending_polygon_move.take() else {
            return false;
        };
        if pending.source_location != self.annotation_location() || pending.start == pending.current
        {
            return false;
        }
        let Some(annotation) = self.annotations.get_mut(pending.annotation_index) else {
            return false;
        };
        if *annotation != pending.original {
            return false;
        }
        *annotation = pending.candidate;
        self.bump_visual_revision();
        true
    }
}
