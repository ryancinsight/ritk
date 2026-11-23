//! Browser host for the same retained framebuffer and event reducer as Windows.
use super::compare::ComparePanel;
use super::layout::{PanelGrid, WorkspaceLayout, MAX_GRID_PANELS};
use super::{NativeViewerObservation, NativeViewerSession, SeriesBrowser};
use crate::app::SnapApp;
use crate::launch::{NativePresentationMode as Mode, NativePresentationSelection as Selection};
use crate::presentation::{PointerButton, PresentationEvent as Event, PresentationModifiers};
use anyhow::{anyhow, Result};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

fn js(error: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&error.to_string()).into()
}

impl NativeViewerSession {
    pub(super) fn open_study_from_dialog(&mut self) -> Result<bool> {
        self.end_held_gestures();
        self.open_requested = true;
        Ok(false)
    }

    pub(super) fn open_selected_series(&mut self, selected: &[usize]) -> Result<bool> {
        let Some((&first, others)) = selected.split_first() else {
            return Ok(false);
        };
        anyhow::ensure!(
            selected.len() <= MAX_GRID_PANELS,
            "too many selected series"
        );
        let browser = self
            .series_browser
            .as_ref()
            .ok_or_else(|| anyhow!("no study catalog"))?;
        let primary = browser.load(first)?;
        let mut panels = arrayvec::ArrayVec::new();
        for (position, &index) in others.iter().enumerate() {
            anyhow::ensure!(
                !selected[..=position].contains(&index),
                "duplicate selected series"
            );
            let mut panel = ComparePanel::empty()?;
            panel.replace(
                browser.load(index)?,
                index,
                "Loaded selected series.".into(),
            )?;
            panels
                .try_push(panel)
                .map_err(|_| anyhow!("too many comparison panels"))?;
        }
        let layout = if others.is_empty() {
            WorkspaceLayout::Orthogonal
        } else {
            let grid = PanelGrid::containing_panel(others.len())
                .ok_or_else(|| anyhow!("invalid panel grid"))?;
            WorkspaceLayout::Panels(grid)
        };
        self.reset_comparison();
        self.app
            .load_volume(primary, "Loaded selected series.".into());
        self.primary_series_index = Some(first);
        self.compare_panels = panels;
        self.set_workspace_layout(layout)?;
        self.active_panel = 0;
        self.series_browser
            .as_mut()
            .expect("validated study")
            .set_active(first);
        Ok(true)
    }

    fn open_payloads(&mut self, batch: metis_web::FileDropBatch) -> Result<()> {
        let files: Vec<_> = batch
            .files()
            .iter()
            .map(|f| (f.name(), f.bytes()))
            .collect();
        if let [(name, bytes)] = files.as_slice()
            && (name.to_ascii_lowercase().ends_with(".nii")
                || name.to_ascii_lowercase().ends_with(".nii.gz"))
        {
            let volume = crate::dicom::loader::load_volume_from_bytes(name, bytes)?;
            self.reset_comparison();
            self.app.load_volume(volume, "Loaded NIfTI volume.".into());
            self.series_browser = None;
            self.primary_series_index = None;
            return Ok(());
        }
        self.open_catalog(SeriesBrowser::from_bytes(&files)?)
    }
}

/// The standalone workstation. The DOM presents pixels and forwards host events.
#[wasm_bindgen]
pub struct AsterSliceWorkspace {
    session: NativeViewerSession,
    dirty: bool,
}

#[wasm_bindgen]
impl AsterSliceWorkspace {
    /// Construct the native workspace, optionally selecting a projection layout.
    #[wasm_bindgen(constructor)]
    pub fn new(layout: &str) -> Result<AsterSliceWorkspace, JsValue> {
        let mode = match layout {
            "" => Selection::Fixed(Mode::Orthogonal),
            "responsive" => Selection::Responsive,
            "mip" => Selection::Fixed(Mode::OrthogonalWithMip),
            "minip" => Selection::Fixed(Mode::OrthogonalWithMinip),
            "average" => Selection::Fixed(Mode::OrthogonalWithAverage),
            "vr" => Selection::Fixed(Mode::OrthogonalWithVr),
            _ => return Err(js("unknown workspace layout")),
        };
        let session = NativeViewerSession::new_with_browser(
            SnapApp::default(),
            Arc::new(NativeViewerObservation::default()),
            false,
            mode,
            false,
            None,
            None,
        )
        .map_err(js)?;
        metis_web::metis_start();
        Ok(Self {
            session,
            dirty: true,
        })
    }

    /// Consume an operating-system file picker request during the trusted event.
    pub fn take_open_request(&mut self) -> bool {
        std::mem::take(&mut self.session.open_requested)
    }

    /// Supply the full client rectangle in CSS pixels.
    pub fn resize(&mut self, width: u32, height: u32, now: f64) -> Result<(), JsValue> {
        self.event(Event::Resized { width, height }, now)
    }

    /// Forward pointer input in framebuffer coordinates, including capture loss.
    pub fn pointer(
        &mut self,
        phase: &str,
        x: f64,
        y: f64,
        button: i16,
        modifiers: u8,
        now: f64,
    ) -> Result<(), JsValue> {
        let button = match button {
            0 => PointerButton::Left,
            1 => PointerButton::Middle,
            2 => PointerButton::Right,
            3 => PointerButton::X1,
            4 => PointerButton::X2,
            _ if phase == "pointermove" => PointerButton::Left,
            _ => return Ok(()),
        };
        let event = match phase {
            "pointerdown" => Event::PointerDown {
                x,
                y,
                button,
                modifiers: keys(modifiers),
            },
            "pointerup" => Event::PointerUp { x, y, button },
            "pointermove" => Event::PointerMove { x, y },
            "dblclick" => Event::PointerDoubleClick { x, y, button },
            "pointercancel" => Event::PointerCancel { x, y, button },
            _ => return Err(js("unknown pointer event")),
        };
        self.event(event, now)
    }

    /// Forward wheel motion in signed native detent units.
    pub fn wheel(
        &mut self,
        x: f64,
        y: f64,
        dx: f64,
        dy: f64,
        modifiers: u8,
        now: f64,
    ) -> Result<(), JsValue> {
        self.event(
            Event::PointerWheel {
                x,
                y,
                delta_x: dx,
                delta_y: dy,
                modifiers: keys(modifiers),
            },
            now,
        )
    }

    /// Forward focus and capture lifecycle.
    pub fn lifecycle(&mut self, phase: &str, now: f64) -> Result<(), JsValue> {
        self.event(
            match phase {
                "focus" => Event::FocusGained,
                "blur" => Event::FocusLost,
                "lostpointercapture" => Event::PointerCaptureLost,
                _ => return Err(js("unknown lifecycle event")),
            },
            now,
        )
    }

    /// Route a keyboard message through the native shortcuts.
    pub fn key(
        &mut self,
        down: bool,
        virtual_key: u32,
        repeated: bool,
        modifiers: u8,
        now: f64,
    ) -> Result<(), JsValue> {
        let modifiers = keys(modifiers);
        self.event(
            if down {
                Event::KeyDown {
                    virtual_key,
                    repeated,
                    modifiers,
                }
            } else {
                Event::KeyUp {
                    virtual_key,
                    modifiers,
                }
            },
            now,
        )
    }

    /// Deliver typed characters to native text controls, including calibration.
    pub fn text(&mut self, text: &str, now: f64) -> Result<(), JsValue> {
        for character in text.chars() {
            self.event(Event::TextInput { character }, now)?;
        }
        Ok(())
    }

    /// Poll completed file transfers and advance the same cine clocks as the desktop.
    pub fn tick(&mut self, now: f64) -> Result<bool, JsValue> {
        let batch = metis_web::take_file_drop();
        let opened = batch.is_some();
        if let Some(batch) = batch
            && let Err(error) = self.session.open_payloads(batch)
        {
            self.session.active_app_mut().status_message =
                format!("Study could not be opened: {error:#}");
        }
        if self.session.tick_cine_at(now) || opened {
            self.session.refresh_frame().map_err(js)?;
            self.dirty = true;
        }
        Ok(std::mem::take(&mut self.dirty))
    }

    /// Return the composed native ARGB framebuffer as Canvas2D RGBA.
    pub fn rgba(&self) -> Vec<u8> {
        self.session
            .framebuffer
            .pixels()
            .iter()
            .flat_map(|pixel| pixel.rotate_left(8).to_be_bytes())
            .collect()
    }

    /// Report identifier-free reading state for host acceptance and accessibility.
    pub fn state(&self) -> String {
        let s = &self.session;
        let app = s.active_app().presentation_snapshot();
        let panes: Vec<_> = s
            .viewports
            .iter()
            .filter(|v| v.panel_width > 0)
            .map(|v| [v.panel_x, v.panel_y, v.panel_width, v.panel_height])
            .collect();
        serde_json::json!({"loaded":app.loaded(),"tool":app.active_tool_name(),"indices":app.slice_indices(),"counts":app.slice_counts(),
            "zoom":app.zoom(),"cine":app.cine_enabled(),"annotations":app.annotation_count(),"panes":panes,
            "measurement":app.last_annotation().map(|a|a.primary_value()),
            "series":s.series_browser.as_ref().map_or(0,SeriesBrowser::len),"activeSeries":s.primary_series_index,
            "positionSync": s.position_sync != super::synchronization::PositionSync::Off, "projection":s.projection.is_some(), "orientationIdentity":app.view_transform().is_identity(),
            "window":[app.window_level().center, app.window_level().width], "crosshair":app.crosshair_visible(),
            "images":s.viewports.iter().map(|v|[v.image.x,v.image.y,v.image.width,v.image.height]).collect::<Vec<_>>(),
            "panels":std::iter::once(&s.app).chain(s.compare_panels.iter().map(|p| &p.app))
                .map(|a| a.axis_slice_info(0)).collect::<Vec<_>>(),
            "panel":s.active_panel,"width":s.framebuffer.width(),"height":s.framebuffer.height()}).to_string()
    }
}

impl AsterSliceWorkspace {
    fn event(&mut self, event: Event, now: f64) -> Result<(), JsValue> {
        match self
            .session
            .reduce_events(Box::new([event]), now)
            .map_err(js)?
        {
            Some(repaint) => self.dirty |= repaint,
            None => {
                let old = &self.session;
                let mut session = NativeViewerSession::new_with_browser(
                    SnapApp::default(),
                    Arc::default(),
                    false,
                    old.presentation_mode,
                    false,
                    None,
                    None,
                )
                .map_err(js)?;
                session
                    .reduce_events(
                        Box::new([Event::Resized {
                            width: old.surface_width,
                            height: old.surface_height,
                        }]),
                        now,
                    )
                    .map_err(js)?;
                self.session = session;
                self.dirty = true;
            }
        }
        Ok(())
    }
}
impl Drop for AsterSliceWorkspace {
    fn drop(&mut self) {
        metis_web::metis_stop();
    }
}

fn keys(bits: u8) -> PresentationModifiers {
    PresentationModifiers::new(bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0)
}
