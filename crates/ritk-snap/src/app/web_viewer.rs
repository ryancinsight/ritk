//! Métis browser canvas viewer workflow.
//!
//! This module is the first browser migration slice beyond the eframe shell:
//! RITK consumes the bounded byte handoff from Métis, applies its existing
//! dropped-input classifier and presents the selected RITK frame through the
//! borrowed canvas seam. DICOM parsing and viewer state stay in [`SnapApp`].

use super::SnapApp;
use crate::app::action_adapter::{ViewerActionDisposition, ViewerViewport};
use crate::presentation::{PresentationFrame, WebCanvasPresenter};
use crate::ui::decide_dropped_input_action;
use crate::ui::ViewTransform;
use moirai_pal::wasm::{spawn_local_with_handle, LocalTaskHandle, WebTimer};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;
use wasm_bindgen::JsValue;

const FRAME_INTERVAL: Duration = Duration::from_millis(16);

thread_local! {
    static VIEWER_TASK: RefCell<Option<LocalTaskHandle>> = const { RefCell::new(None) };
}

struct BrowserViewer {
    app: SnapApp,
    surface: BrowserSurface,
}

impl BrowserViewer {
    fn new(surface: BrowserSurface) -> Self {
        Self {
            app: SnapApp::default(),
            surface,
        }
    }

    fn tick(&mut self) -> std::io::Result<bool> {
        let mut repaint = false;
        let dropped = super::browser_input::take_dropped_files();
        if !dropped.is_empty() {
            let action = decide_dropped_input_action(&dropped);
            self.app.apply_dropped_input_action(action);
            self.surface.clear();
            repaint = true;
        }
        let disposition = self.surface.apply_events(&mut self.app)?;
        let ViewerActionDisposition::Continue {
            repaint: input_repaint,
        } = disposition
        else {
            return Ok(false);
        };
        if repaint || input_repaint {
            self.surface.clear();
        }
        self.surface.render_and_present(&self.app)?;
        Ok(true)
    }
}

enum BrowserSurface {
    Single {
        presenter: WebCanvasPresenter,
        frame: Option<PresentationFrame>,
    },
    Orthogonal {
        presenters: [WebCanvasPresenter; 3],
        frames: Option<[PresentationFrame; 3]>,
    },
}

impl BrowserSurface {
    fn clear(&mut self) {
        match self {
            Self::Single { frame, .. } => *frame = None,
            Self::Orthogonal { frames, .. } => *frames = None,
        }
    }

    fn render_and_present(&mut self, app: &SnapApp) -> std::io::Result<()> {
        match self {
            Self::Single { presenter, frame } => {
                if frame.is_none() {
                    *frame = app
                        .render_browser_frame()
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                }
                if let Some(frame) = frame.as_ref() {
                    presenter.present(frame)?;
                }
            }
            Self::Orthogonal { presenters, frames } => {
                if frames.is_none() {
                    *frames = app
                        .render_browser_frames()
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                }
                if let Some(frames) = frames.as_ref() {
                    for (presenter, frame) in presenters.iter().zip(frames) {
                        presenter.present(frame)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn apply_events(&mut self, app: &mut SnapApp) -> std::io::Result<ViewerActionDisposition> {
        match self {
            Self::Single { presenter, frame } => {
                apply_canvas_events(app, 0, presenter, frame.as_ref())
            }
            Self::Orthogonal { presenters, frames } => {
                let mut repaint = false;
                for (axis, presenter) in presenters.iter_mut().enumerate() {
                    let frame = frames.as_ref().and_then(|frames| frames.get(axis));
                    match apply_canvas_events(app, axis, presenter, frame)? {
                        ViewerActionDisposition::Continue { repaint: needed } => {
                            repaint |= needed;
                        }
                        ViewerActionDisposition::Exit => {
                            return Ok(ViewerActionDisposition::Exit);
                        }
                    }
                }
                Ok(ViewerActionDisposition::Continue { repaint })
            }
        }
    }
}

fn apply_canvas_events(
    app: &mut SnapApp,
    axis: usize,
    presenter: &mut WebCanvasPresenter,
    frame: Option<&PresentationFrame>,
) -> std::io::Result<ViewerActionDisposition> {
    let events = match presenter.take_events() {
        Ok(events) => events,
        Err(error) => {
            app.cancel_presentation_gesture();
            return Err(std::io::Error::other(error.to_string()));
        }
    };
    if events.is_empty() {
        return Ok(ViewerActionDisposition::Continue { repaint: false });
    }
    let viewport = frame
        .map(|frame| viewport_for_frame(axis, frame))
        .transpose()?;
    let previous_axis = app.axis;
    app.axis = axis;
    let disposition = match app.apply_presentation_events(&events, viewport.as_ref()) {
        Ok(disposition) => disposition,
        Err(error) => {
            app.axis = previous_axis;
            app.cancel_presentation_gesture();
            return Err(std::io::Error::other(format!(
                "apply RITK browser presentation events: {error}"
            )));
        }
    };
    Ok(disposition)
}

fn viewport_for_frame(axis: usize, frame: &PresentationFrame) -> std::io::Result<ViewerViewport> {
    let width = usize::try_from(frame.width())
        .map_err(|_| std::io::Error::other("browser frame width exceeds host range"))?;
    let height = usize::try_from(frame.height())
        .map_err(|_| std::io::Error::other("browser frame height exceeds host range"))?;
    ViewerViewport::new(
        axis,
        egui::pos2(0.0, 0.0),
        egui::vec2(1.0, 1.0),
        [width, height],
        ViewTransform::default(),
    )
    .map_err(|error| std::io::Error::other(error.to_string()))
}

fn launch_browser_viewer(viewer: BrowserViewer) -> Result<(), JsValue> {
    let viewer = Rc::new(RefCell::new(viewer));
    let task_viewer = Rc::clone(&viewer);
    let handle = spawn_local_with_handle(async move {
        loop {
            let keep_running = match task_viewer.borrow_mut().tick() {
                Ok(keep_running) => keep_running,
                Err(error) => {
                    tracing::error!(%error, "RITK browser canvas workflow stopped");
                    break;
                }
            };
            if !keep_running {
                break;
            }
            let timer = match WebTimer::new(FRAME_INTERVAL) {
                Ok(timer) => timer,
                Err(error) => {
                    tracing::error!(%error, "RITK browser canvas timer stopped");
                    break;
                }
            };
            if let Err(error) = timer.await {
                tracing::error!(%error, "RITK browser canvas timer stopped");
                break;
            }
        }
    });
    VIEWER_TASK.with_borrow_mut(|slot| slot.replace(handle));
    Ok(())
}

/// Starts the RITK DICOM byte-drop workflow on a Métis-owned browser canvas.
pub(crate) fn start_web_canvas(canvas_id: String) -> Result<(), JsValue> {
    stop_web_canvas();
    let presenter = match WebCanvasPresenter::from_canvas_id_with_input(&canvas_id) {
        Ok(presenter) => presenter,
        Err(error) => return Err(JsValue::from_str(&error.to_string())),
    };
    metis_web::metis_start();
    launch_browser_viewer(BrowserViewer::new(BrowserSurface::Single {
        presenter,
        frame: None,
    }))
}

/// Starts the RITK DICOM byte-drop workflow on three Métis-owned canvases.
pub(crate) fn start_web_orthogonal_canvases(canvas_ids: [String; 3]) -> Result<(), JsValue> {
    stop_web_canvas();
    let [axial_id, coronal_id, sagittal_id] = canvas_ids;
    let presenters = match (
        WebCanvasPresenter::from_canvas_id_with_input(&axial_id),
        WebCanvasPresenter::from_canvas_id_with_input(&coronal_id),
        WebCanvasPresenter::from_canvas_id_with_input(&sagittal_id),
    ) {
        (Ok(axial), Ok(coronal), Ok(sagittal)) => [axial, coronal, sagittal],
        (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => {
            return Err(JsValue::from_str(&error.to_string()));
        }
    };
    metis_web::metis_start();
    launch_browser_viewer(BrowserViewer::new(BrowserSurface::Orthogonal {
        presenters,
        frames: None,
    }))
}

/// Stops the RITK browser canvas workflow and releases its timer task.
pub(crate) fn stop_web_canvas() {
    VIEWER_TASK.with_borrow_mut(|slot| slot.take());
    metis_web::metis_stop();
}

impl SnapApp {
    fn render_browser_frame(&self) -> anyhow::Result<Option<PresentationFrame>> {
        let Some(volume) = self.loaded.as_ref() else {
            return Ok(None);
        };
        let window_level = self.browser_window_level();
        let slice_index = match self.axis {
            0 => self.viewer_state.slice_index,
            1 => self.coronal_slice,
            _ => self.sagittal_slice,
        };
        PresentationFrame::from_slice(volume, self.axis, slice_index, window_level, self.colormap)
            .map(Some)
    }

    fn render_browser_frames(&self) -> anyhow::Result<Option<[PresentationFrame; 3]>> {
        let Some(volume) = self.loaded.as_ref() else {
            return Ok(None);
        };
        let indices = [0_usize, 1, 2].map(|axis| self.axis_slice_info(axis).0);
        PresentationFrame::from_orthogonal_slices(
            volume,
            indices,
            self.browser_window_level(),
            self.colormap,
        )
        .map(Some)
    }

    fn browser_window_level(&self) -> crate::render::WindowLevel {
        let window_center = self
            .viewer_state
            .window_center
            .unwrap_or(crate::viewer::DEFAULT_WINDOW_CENTER);
        let window_width = self
            .viewer_state
            .window_width
            .unwrap_or(crate::viewer::DEFAULT_WINDOW_WIDTH)
            .max(1.0);
        crate::render::WindowLevel::new(f64::from(window_center), f64::from(window_width))
    }
}
