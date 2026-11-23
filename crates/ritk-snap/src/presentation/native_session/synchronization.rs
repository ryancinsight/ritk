//! Apply a user-selected slice to eligible visible comparison panels.
use super::NativeViewerSession;
use crate::app::SnapApp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PositionSync {
    Automatic,
    Manual,
    Off,
}

impl NativeViewerSession {
    pub(super) fn toggle_position_sync(&mut self) {
        self.position_sync = match self.position_sync {
            PositionSync::Off => PositionSync::Automatic,
            PositionSync::Automatic | PositionSync::Manual => PositionSync::Off,
        };
        self.active_app_mut().status_message = if self.position_sync == PositionSync::Off {
            "Patient-position synchronization disabled"
        } else {
            "Automatic patient-position synchronization enabled"
        }
        .into();
    }

    pub(super) fn align_current_slices(&mut self) {
        let count = self.workspace_layout.grid().map_or(0, |g| g.panel_count());
        let source = self.active_app().patient_stack.as_ref();
        let eligible = std::iter::once(&self.app)
            .chain(self.compare_panels.iter().map(|p| &p.app))
            .take(count)
            .enumerate()
            .any(|(i, app)| {
                i != self.active_panel
                    && source
                        .zip(app.patient_stack.as_ref())
                        .is_some_and(|(s, t)| s.can_anchor(t))
            });
        if !eligible {
            self.active_app_mut().status_message =
                "Manual alignment needs two positioned stacks within 30 degrees".into();
            return;
        }
        for (i, app) in std::iter::once(&mut self.app)
            .chain(self.compare_panels.iter_mut().map(|p| &mut p.app))
            .enumerate()
        {
            let index = app.axis_slice_info(0).0;
            if let Some(stack) = &mut app.patient_stack {
                stack.anchor = (i < count).then_some(index);
            }
        }
        self.position_sync = PositionSync::Manual;
        self.active_app_mut().status_message = "Manual alignment set at the selected slices".into();
    }

    pub(super) fn synchronize_position(&mut self, source: usize) {
        if self.position_sync == PositionSync::Off {
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
            if let Some(next) =
                target
                    .patient_stack
                    .as_ref()
                    .and_then(|t| match self.position_sync {
                        PositionSync::Automatic => stack.corresponding_slice(index, t),
                        PositionSync::Manual => stack.anchored_slice(index, t),
                        PositionSync::Off => None,
                    })
            {
                target.set_slice_for_axis(0, next);
            }
        }
    }
}
