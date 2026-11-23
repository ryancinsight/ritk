//! Apply a user-selected slice to eligible visible comparison panels.
use super::NativeViewerSession;
use crate::app::SnapApp;

impl NativeViewerSession {
    pub(super) fn synchronize_position(&mut self, source: usize) {
        if !self.sync_position {
            return;
        }
        let Some(grid) = self.workspace_layout.grid() else {
            return;
        };
        let count = grid.panel_count();
        let mut apps = std::iter::once(&mut self.app)
            .chain(self.compare_panels.iter_mut().map(|p| &mut p.app))
            .take(count)
            .collect::<arrayvec::ArrayVec<_, { super::layout::MAX_GRID_PANELS }>>();
        let Some((source, before, after)) = apps
            .split_at_mut_checked(source)
            .and_then(|(before, tail)| tail.split_first_mut().map(|(s, after)| (s, before, after)))
        else {
            return;
        };
        let source: &SnapApp = source;
        let Some(stack) = source.patient_stack.as_ref() else {
            return;
        };
        let index = source.axis_slice_info(0).0;
        for target in before.iter_mut().chain(after) {
            if let Some(next) = target
                .patient_stack
                .as_ref()
                .and_then(|t| stack.corresponding_slice(index, t))
            {
                target.set_slice_for_axis(0, next);
            }
        }
    }
}
