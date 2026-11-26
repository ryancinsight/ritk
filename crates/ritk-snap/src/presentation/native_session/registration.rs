//! Windows Tools action for rigid registration of the first two displayed series.
use super::compare::ComparePanel;
use super::layout::{PanelGrid, WorkspaceLayout, MAX_GRID_PANELS};
use super::NativeViewerSession;
use crate::geometry::patient_stack::PatientStack;
use crate::registration::{run, scalar_input};
use crate::LoadedVolume;
use anyhow::{ensure, Context, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

pub(super) struct RegistrationJob {
    inputs: [Weak<Vec<f32>>; 2],
    _cancel_on_drop: CancelOnDrop,
    handle: moirai::TaskHandle<Result<LoadedVolume>>,
}

struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl NativeViewerSession {
    fn registration_inputs(&self) -> Option<[&LoadedVolume; 2]> {
        let grid = self.workspace_layout.grid()?;
        let second = self.compare_panels.first()?;
        if grid.panel_count() < 2
            || self.app.primary_load_active()
            || second.app.primary_load_active()
        {
            return None;
        }
        Some([self.app.loaded.as_ref()?, second.app.loaded.as_ref()?])
    }

    pub(super) fn start_registration(&mut self) -> Result<()> {
        ensure!(
            self.registration.is_none(),
            "Registration is running. Press Esc to cancel."
        );
        let [fixed, moving] = self
            .registration_inputs()
            .context("Use F4 to open the fixed series in P1 and the moving series in P2.")?;
        let pair = fixed.metadata.as_deref().zip(moving.metadata.as_deref());
        let same_series = pair
            .and_then(|(a, b)| {
                a.series_instance_uid
                    .as_ref()
                    .zip(b.series_instance_uid.as_ref())
            })
            .is_some_and(|(a, b)| a == b);
        ensure!(
            !Arc::ptr_eq(&fixed.data, &moving.data) && !same_series,
            "Open two distinct series."
        );
        let shared_frame = pair
            .and_then(|(a, b)| {
                a.common_spatial_reference()
                    .zip(b.common_spatial_reference())
            })
            .is_some_and(|(a, b)| a.1 == b.1);
        let inputs = [Arc::downgrade(&fixed.data), Arc::downgrade(&moving.data)];
        let fixed = scalar_input(fixed)?;
        let moving = scalar_input(moving)?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = Arc::clone(&cancelled);
        let handle = moirai::global()
            .spawn_blocking(move || run(&fixed, &moving, shared_frame, &worker_cancelled));
        self.registration = Some(RegistrationJob {
            inputs,
            handle,
            _cancel_on_drop: CancelOnDrop(cancelled),
        });
        self.active_app_mut().status_message = "Registering P2 to P1. Esc cancels.".into();
        Ok(())
    }

    pub(super) fn poll_registration(&mut self) -> bool {
        let Some(job) = self.registration.as_ref() else {
            return false;
        };
        let current = self.registration_inputs().is_some_and(|volumes| {
            job.inputs
                .iter()
                .zip(volumes)
                .all(|(key, volume)| key.ptr_eq(&Arc::downgrade(&volume.data)))
        });
        if !current {
            self.registration = None;
            self.active_app_mut().status_message =
                "Registration cancelled because the input panels changed.".into();
            return true;
        }
        if !job.handle.is_finished() {
            return false;
        }
        let Some(job) = self.registration.take() else {
            return false;
        };
        let result = job
            .handle
            .join()
            .context("registration worker returned no result")
            .and_then(|result| Ok(result??))
            .and_then(|output| self.show_registration(output));
        if let Err(error) = result {
            self.active_app_mut().status_message = format!("Registration failed: {error:#}");
        }
        true
    }

    fn show_registration(&mut self, volume: LoadedVolume) -> Result<()> {
        let fixed = self.app.loaded.as_ref().context("fixed volume closed")?;
        let mut preview = ComparePanel::empty()?;
        preview.replace_volume(
            volume,
            None,
            "Rigid registration preview. Check corresponding landmarks in all three planes.".into(),
        )?;
        // Geometry for slice linking comes from the grid onto which the moving
        // image was resampled. The preview has no source DICOM catalog entry.
        preview.app.patient_stacks = PatientStack::displayed_planes(fixed);
        preview.app.axis = self.app.axis;
        preview
            .app
            .set_slice_for_axis(self.app.axis, self.app.axis_slice_info(self.app.axis).0);
        let target = self
            .compare_panels
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, p)| {
                p.series_index.is_none()
                    && !p.app.primary_load_active()
                    && p.app
                        .loaded
                        .as_ref()
                        .is_none_or(LoadedVolume::is_registration_preview)
            })
            .map_or(self.compare_panels.len() + 1, |(index, _)| index + 1);
        ensure!(
            target < MAX_GRID_PANELS,
            "Close a comparison panel to display the registered series."
        );
        if self
            .workspace_layout
            .grid()
            .is_none_or(|grid| grid.panel_count() <= target)
        {
            let grid =
                PanelGrid::containing_panel(target).context("no space for registration preview")?;
            self.set_workspace_layout(WorkspaceLayout::Panels(grid))?;
        }
        self.compare_panels[target - 1] = preview;
        self.active_panel = target;
        Ok(())
    }
}
