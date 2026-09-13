#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use crate::ui::advance_wrapped;
use crate::ui::{axis_total, clamp_index, step_clamped};
#[cfg(not(target_arch = "wasm32"))]
use crate::LoadedVolume;

use super::state::SnapApp;

impl SnapApp {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn axis_extent_for_volume(volume: &LoadedVolume, axis: usize) -> usize {
        match axis {
            0 => volume.shape[0],
            1 => volume.shape[1],
            _ => volume.shape[2],
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn map_slice_index_between_volumes(
        primary_index: usize,
        primary_total: usize,
        secondary_total: usize,
    ) -> usize {
        if primary_total <= 1 || secondary_total <= 1 {
            return 0;
        }
        let pmax = primary_total.saturating_sub(1) as f64;
        let smax = secondary_total.saturating_sub(1) as f64;
        ((primary_index as f64 / pmax) * smax)
            .round()
            .clamp(0.0, smax) as usize
    }

    // ── Slice navigation ──────────────────────────────────────────────────────

    /// Return `(current_slice_index, total_slices)` for `axis`.
    pub(crate) fn axis_slice_info(&self, axis: usize) -> (usize, usize) {
        let total = self
            .loaded
            .as_ref()
            .map(|v| axis_total(v.shape, axis))
            .unwrap_or(1);
        match axis {
            0 => (self.viewer_state.slice_index, total),
            1 => (self.coronal_slice, total),
            _ => (self.sagittal_slice, total),
        }
    }

    /// Step the slice for `axis` by `delta`, clamped to the valid range.
    ///
    /// Advances the visual revision when the index changes.
    pub(crate) fn set_slice_for_axis(&mut self, axis: usize, index: usize) {
        let total = self
            .loaded
            .as_ref()
            .map(|v| axis_total(v.shape, axis))
            .unwrap_or(1);
        let next = clamp_index(index, total);
        match axis {
            0 => {
                if next != self.viewer_state.slice_index {
                    self.viewer_state.slice_index = next;
                    self.bump_visual_revision();
                    if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor.as_mut()) {
                        cursor.set_axis_slice(vol.shape, 0, next);
                    }
                }
            }
            1 => {
                if next != self.coronal_slice {
                    self.coronal_slice = next;
                    self.bump_visual_revision();
                    if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor.as_mut()) {
                        cursor.set_axis_slice(vol.shape, 1, next);
                    }
                }
            }
            _ => {
                if next != self.sagittal_slice {
                    self.sagittal_slice = next;
                    self.bump_visual_revision();
                    if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor.as_mut()) {
                        cursor.set_axis_slice(vol.shape, 2, next);
                    }
                }
            }
        }
    }

    /// Step the slice for `axis` by `delta`, clamped to the valid range.
    ///
    /// Advances the visual revision when the index changes.
    pub(crate) fn step_slice_for_axis(&mut self, axis: usize, delta: i32) {
        let (current, total) = self.axis_slice_info(axis);
        let next = step_clamped(current, total, delta);
        self.set_slice_for_axis(axis, next);
    }

    /// Step the primary-axis slice by `delta`. Delegates to
    /// [`step_slice_for_axis`] using `self.axis`.
    ///
    /// [`step_slice_for_axis`]: SnapApp::step_slice_for_axis
    pub(crate) fn step_slice(&mut self, delta: i32) {
        self.step_slice_for_axis(self.axis, delta);
    }

    /// Apply the keyboard navigation transitions shared by native and browser hosts.
    pub(crate) fn apply_slice_navigation_shortcuts(
        &mut self,
        arrow_up: bool,
        arrow_down: bool,
        page_up: bool,
        page_down: bool,
        home: bool,
        end: bool,
    ) {
        if arrow_up || page_up {
            self.step_slice(-1);
        } else if arrow_down || page_down {
            self.step_slice(1);
        } else if home {
            self.jump_active_axis_slice_boundary(false);
        } else if end {
            self.jump_active_axis_slice_boundary(true);
        }
    }

    fn jump_active_axis_slice_boundary(&mut self, end: bool) {
        let (_, total) = self.axis_slice_info(self.axis);
        let target = if end { total.saturating_sub(1) } else { 0 };
        self.set_slice_for_axis(self.axis, target);
    }

    /// Advance `axis` by `steps` with wrap-around.
    ///
    /// Delegates the actual write to [`set_slice_for_axis`] so visual revision,
    /// linked-cursor synchronisation, and the no-change guard are all applied
    /// through the shared state path.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn advance_slice_for_axis_loop(&mut self, axis: usize, steps: u32) {
        if steps == 0 {
            return;
        }
        let total = self
            .loaded
            .as_ref()
            .map(|v| axis_total(v.shape, axis))
            .unwrap_or(1);
        if total == 0 {
            return;
        }
        let current = match axis {
            0 => self.viewer_state.slice_index,
            1 => self.coronal_slice,
            _ => self.sagittal_slice,
        };
        let next = advance_wrapped(current, total, steps);
        self.set_slice_for_axis(axis, next);
    }

    /// Advance cine playback for the active axis and schedule repaints.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn tick_cine(&mut self, ctx: &egui::Context) {
        if self.loaded.is_none() {
            self.cine.stop();
            return;
        }
        if !self.cine.enabled {
            return;
        }
        let now = ctx.input(|i| i.time);
        let steps = self.cine.consume_steps(now);
        if steps > 0 {
            self.advance_slice_for_axis_loop(self.axis, steps);
            ctx.request_repaint();
        } else {
            ctx.request_repaint_after(Duration::from_millis(8));
        }
    }
}
