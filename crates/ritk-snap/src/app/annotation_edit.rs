use super::pointer_bindings::{AnnotationMoveKind, PendingAnnotationMove};
use super::state::SnapApp;
use super::viewer_viewport::ViewerViewport;
use crate::presentation::ViewportPoint;
use crate::tools::interaction::ImagePoint;
use crate::tools::kind::ToolKind;

mod length;
mod polygon;

impl SnapApp {
    pub(crate) fn begin_annotation_move(
        &mut self,
        viewport: ViewerViewport,
        screen: ViewportPoint,
        point: ImagePoint,
        tool: ToolKind,
    ) -> bool {
        if !matches!(tool, ToolKind::Pan | ToolKind::MeasureLength) {
            return false;
        }
        let selected = self.pointer.selected_annotation;
        let location = self.annotation_location();
        let spacing = self.slice_plane_spacing().ok();
        let pending = self
            .annotations
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, annotation)| annotation.location() == Some(location))
            .find_map(|(annotation_index, original)| {
                let (kind, candidate) = if let Some(mode) = length::pick(original, viewport, screen)
                {
                    (AnnotationMoveKind::Length(mode), original.clone())
                } else if tool == ToolKind::Pan {
                    if let Some(node_index) = polygon::nearest_node(original, viewport, screen) {
                        (
                            AnnotationMoveKind::Polygon {
                                node_index,
                                inserted: false,
                            },
                            original.clone(),
                        )
                    } else {
                        let (edge, fraction) = polygon::nearest_edge(original, viewport, screen)?;
                        if selected != Some(annotation_index) {
                            return Some(None);
                        }
                        let spacing = spacing?;
                        let candidate = polygon::insert_node(original, edge, fraction, spacing)?;
                        (
                            AnnotationMoveKind::Polygon {
                                node_index: edge + 1,
                                inserted: true,
                            },
                            candidate,
                        )
                    }
                } else {
                    return None;
                };
                Some(Some(PendingAnnotationMove {
                    annotation_index,
                    source_location: location,
                    original: original.clone(),
                    kind,
                    candidate,
                    start: point,
                    current: point,
                }))
            })
            .flatten();
        let Some(pending) = pending else {
            return false;
        };
        self.pointer.selected_annotation = Some(pending.annotation_index);
        self.pointer.pending_annotation_move = Some(pending);
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
        let Some(candidate) = (match pending.kind {
            AnnotationMoveKind::Length(mode) => {
                length::move_candidate(&pending.original, mode, point, spacing)
            }
            AnnotationMoveKind::Polygon {
                node_index,
                inserted,
            } => {
                let source = if inserted {
                    &pending.candidate
                } else {
                    &pending.original
                };
                polygon::move_node(source, node_index, point, spacing)
            }
        }) else {
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
            && let Some(annotation) = self.annotations.get_mut(pending.annotation_index)
            && *annotation == pending.original
        {
            self.pointer.selected_annotation = Some(pending.annotation_index);
            if matches!(
                pending.kind,
                AnnotationMoveKind::Polygon { inserted: true, .. }
            ) || pending.start != pending.current
            {
                *annotation = pending.candidate;
                self.bump_visual_revision();
            }
        }
    }
}
