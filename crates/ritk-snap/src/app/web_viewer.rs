//! Métis browser canvas viewer workflow.
//!
//! This module is the first browser migration slice beyond the eframe shell:
//! RITK consumes the bounded byte handoff from Métis, applies its existing
//! dropped-input classifier and presents the selected RITK frame through the
//! borrowed canvas seam. It also publishes a bounded semantic snapshot on
//! each canvas for consumer-owned workflow assertions. DICOM parsing and
//! viewer state stay in [`SnapApp`].

use super::browser_geometry::{viewport_for_display, PhysicalCanvasAspect};
use super::browser_semantics::BrowserCanvasSemantics;
use super::browser_slice_selection::{parse_browser_slice_request, BrowserSliceSelectionError};
use super::SnapApp;
use crate::app::action_adapter::ViewerActionDisposition;
use crate::presentation::{PresentationFrame, WebCanvasPresenter};
use crate::ui::decide_dropped_input_action;
use moirai_pal::wasm::{
    spawn_local_with_handle, LocalTaskHandle, WebAnimationFrame, WebDocument, WebElement,
};
use std::cell::RefCell;
use wasm_bindgen::JsValue;

const MILLISECONDS_PER_SECOND: f64 = 1_000.0;

thread_local! {
    static VIEWER_TASK: RefCell<Option<LocalTaskHandle>> = const { RefCell::new(None) };
    static VIEWER: RefCell<Option<BrowserViewer>> = const { RefCell::new(None) };
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

    fn tick(&mut self, now_seconds: f64) -> std::io::Result<bool> {
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
        if matches!(
            self.app.tick_cine_at(now_seconds),
            super::slice_ops::CineTick::Advanced(_)
        ) {
            self.surface.clear();
        }
        self.surface.render_and_present(&self.app)?;
        self.surface.publish_semantics(&self.app)?;
        Ok(true)
    }
}

struct BrowserCanvas {
    presenter: WebCanvasPresenter,
    element: WebElement,
    last_semantics: Option<BrowserCanvasSemantics>,
    last_physical_aspect: Option<PhysicalCanvasAspect>,
    frame_generation: u64,
}

impl BrowserCanvas {
    fn from_id(id: &str) -> std::io::Result<Self> {
        let presenter = WebCanvasPresenter::from_canvas_id_with_input(id)?;
        let document = WebDocument::current()?;
        let element = document.get_element_by_id(id).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "browser canvas element disappeared during setup",
            )
        })?;
        // Keyboard events target the focused canvas; make the retained
        // presentation surface keyboard-focusable at the RITK boundary.
        element.set_attribute("tabindex", "0")?;
        Ok(Self {
            presenter,
            element,
            last_semantics: None,
            last_physical_aspect: None,
            frame_generation: 0,
        })
    }

    /// Counts newly rendered frames only after their canvas upload succeeds.
    /// Cached animation-frame uploads do not establish repaint evidence.
    fn present_rendered_frame(&mut self, frame: &PresentationFrame) -> std::io::Result<()> {
        let generation = self
            .frame_generation
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("browser rendered-frame generation exhausted"))?;
        self.presenter.present(frame)?;
        self.element
            .set_attribute("data-ritk-frame-generation", &generation.to_string())?;
        self.frame_generation = generation;
        Ok(())
    }

    fn publish_semantics(
        &mut self,
        semantics: BrowserCanvasSemantics,
        physical_aspect: Option<PhysicalCanvasAspect>,
    ) -> std::io::Result<()> {
        self.publish_physical_aspect(physical_aspect)?;
        if self.last_semantics == Some(semantics) {
            return Ok(());
        }
        let axis = semantics.axis.to_string();
        let slice_index = semantics.slice_index.to_string();
        let slice_count = semantics.slice_count.to_string();
        let (width, height) = semantics.frame_dimensions_or_zero();
        let width = width.to_string();
        let height = height.to_string();
        let cine_fps = semantics.cine_fps_value();
        self.element
            .set_attribute("data-ritk-load-state", semantics.load_state_value())?;
        self.element
            .set_attribute("data-ritk-frame-state", semantics.frame_state_value())?;
        self.element.set_attribute("data-ritk-axis", &axis)?;
        self.element
            .set_attribute("data-ritk-slice-index", &slice_index)?;
        self.element
            .set_attribute("data-ritk-slice-count", &slice_count)?;
        self.element
            .set_attribute("data-ritk-frame-width", &width)?;
        self.element
            .set_attribute("data-ritk-frame-height", &height)?;
        self.element
            .set_attribute("data-ritk-cine-fps", &cine_fps)?;
        self.last_semantics = Some(semantics);
        Ok(())
    }

    fn publish_physical_aspect(
        &mut self,
        physical_aspect: Option<PhysicalCanvasAspect>,
    ) -> std::io::Result<()> {
        let Some(physical_aspect) = physical_aspect else {
            return Ok(());
        };
        if self.last_physical_aspect == Some(physical_aspect) {
            return Ok(());
        }
        let value = physical_aspect.attribute_value();
        self.element.set_style_property("width", "100%")?;
        self.element.set_style_property("height", "auto")?;
        self.element.set_style_property("aspect-ratio", &value)?;
        self.element
            .set_attribute("data-ritk-display-aspect", &value)?;
        self.last_physical_aspect = Some(physical_aspect);
        Ok(())
    }
}

enum BrowserSurface {
    Single {
        canvas: BrowserCanvas,
        frame: Option<PresentationFrame>,
    },
    Orthogonal {
        canvases: Box<[BrowserCanvas; 3]>,
        frames: Option<[PresentationFrame; 3]>,
    },
}

impl BrowserSurface {
    fn listener_count(&self) -> usize {
        match self {
            Self::Single { canvas, .. } => canvas.presenter.listener_count(),
            Self::Orthogonal { canvases, .. } => canvases
                .iter()
                .map(|canvas| canvas.presenter.listener_count())
                .sum(),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::Single { frame, .. } => *frame = None,
            Self::Orthogonal { frames, .. } => *frames = None,
        }
    }

    fn render_and_present(&mut self, app: &SnapApp) -> std::io::Result<()> {
        match self {
            Self::Single { canvas, frame } => {
                let rendered = frame.is_none();
                if frame.is_none() {
                    *frame = app
                        .render_browser_frame()
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                }
                if let Some(frame) = frame.as_ref() {
                    if rendered {
                        canvas.present_rendered_frame(frame)?;
                    } else {
                        canvas.presenter.present(frame)?;
                    }
                }
            }
            Self::Orthogonal { canvases, frames } => {
                let rendered = frames.is_none();
                if frames.is_none() {
                    *frames = app
                        .render_browser_frames()
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                }
                if let Some(frames) = frames.as_ref() {
                    for (canvas, frame) in canvases.iter_mut().zip(frames) {
                        if rendered {
                            canvas.present_rendered_frame(frame)?;
                        } else {
                            canvas.presenter.present(frame)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn publish_semantics(&mut self, app: &SnapApp) -> std::io::Result<()> {
        match self {
            Self::Single { canvas, frame } => {
                let axis = app.axis;
                let (slice_index, slice_count) = app.axis_slice_info(axis);
                let semantics = BrowserCanvasSemantics::from_state(
                    app.loaded.is_some(),
                    axis,
                    slice_index,
                    slice_count,
                    frame.as_ref(),
                    app.cine.fps,
                );
                let physical_aspect = physical_aspect(app, axis, frame.as_ref())?;
                canvas.publish_semantics(semantics, physical_aspect)
            }
            Self::Orthogonal { canvases, frames } => {
                for (axis, canvas) in canvases.iter_mut().enumerate() {
                    let frame = frames.as_ref().and_then(|frames| frames.get(axis));
                    let (slice_index, slice_count) = app.axis_slice_info(axis);
                    let semantics = BrowserCanvasSemantics::from_state(
                        app.loaded.is_some(),
                        axis,
                        slice_index,
                        slice_count,
                        frame,
                        app.cine.fps,
                    );
                    let physical_aspect = physical_aspect(app, axis, frame)?;
                    canvas.publish_semantics(semantics, physical_aspect)?;
                }
                Ok(())
            }
        }
    }

    fn apply_events(&mut self, app: &mut SnapApp) -> std::io::Result<ViewerActionDisposition> {
        match self {
            Self::Single { canvas, frame } => apply_canvas_events(app, 0, canvas, frame.as_ref()),
            Self::Orthogonal { canvases, frames } => {
                let mut repaint = false;
                for (axis, canvas) in canvases.iter_mut().enumerate() {
                    let frame = frames.as_ref().and_then(|frames| frames.get(axis));
                    match apply_canvas_events(app, axis, canvas, frame)? {
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

fn physical_aspect(
    app: &SnapApp,
    axis: usize,
    frame: Option<&PresentationFrame>,
) -> std::io::Result<Option<PhysicalCanvasAspect>> {
    let (Some(volume), Some(frame)) = (app.loaded.as_ref(), frame) else {
        return Ok(None);
    };
    PhysicalCanvasAspect::new(volume.spacing, axis, frame.width(), frame.height()).map(Some)
}

fn apply_canvas_events(
    app: &mut SnapApp,
    axis: usize,
    canvas: &mut BrowserCanvas,
    frame: Option<&PresentationFrame>,
) -> std::io::Result<ViewerActionDisposition> {
    let events = match canvas.presenter.take_events() {
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
        .map(|frame| viewport_for_display(axis, [1.0, 1.0], [frame.width(), frame.height()]))
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

fn launch_browser_viewer(viewer: BrowserViewer) -> Result<(), JsValue> {
    VIEWER.with_borrow_mut(|slot| *slot = Some(viewer));
    let handle = spawn_local_with_handle(async move {
        loop {
            let frame = match WebAnimationFrame::new() {
                Ok(frame) => frame,
                Err(error) => {
                    tracing::error!(%error, "RITK browser animation-frame scheduling stopped");
                    stop_web_canvas();
                    break;
                }
            };
            let timestamp_ms = match frame.await {
                Ok(timestamp_ms) => timestamp_ms,
                Err(error) => {
                    tracing::error!(%error, "RITK browser animation-frame wait stopped");
                    stop_web_canvas();
                    break;
                }
            };
            let tick = VIEWER.with_borrow_mut(|slot| {
                let Some(viewer) = slot.as_mut() else {
                    return Ok(false);
                };
                viewer.tick(timestamp_ms / MILLISECONDS_PER_SECOND)
            });
            let keep_running = match tick {
                Ok(keep_running) => keep_running,
                Err(error) => {
                    tracing::error!(%error, "RITK browser canvas workflow stopped");
                    // Teardown drops the viewer synchronously before cancelling
                    // this task, so remount cannot overlap listener generations.
                    stop_web_canvas();
                    break;
                }
            };
            if !keep_running {
                stop_web_canvas();
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
    let canvas = match BrowserCanvas::from_id(&canvas_id) {
        Ok(canvas) => canvas,
        Err(error) => return Err(JsValue::from_str(&error.to_string())),
    };
    metis_web::metis_start();
    launch_browser_viewer(BrowserViewer::new(BrowserSurface::Single {
        canvas,
        frame: None,
    }))
}

/// Starts the RITK DICOM byte-drop workflow on three Métis-owned canvases.
pub(crate) fn start_web_orthogonal_canvases(canvas_ids: [String; 3]) -> Result<(), JsValue> {
    stop_web_canvas();
    let [axial_id, coronal_id, sagittal_id] = canvas_ids;
    let canvases = match (
        BrowserCanvas::from_id(&axial_id),
        BrowserCanvas::from_id(&coronal_id),
        BrowserCanvas::from_id(&sagittal_id),
    ) {
        (Ok(axial), Ok(coronal), Ok(sagittal)) => [axial, coronal, sagittal],
        (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => {
            return Err(JsValue::from_str(&error.to_string()));
        }
    };
    metis_web::metis_start();
    launch_browser_viewer(BrowserViewer::new(BrowserSurface::Orthogonal {
        canvases: Box::new(canvases),
        frames: None,
    }))
}

/// Stops the RITK browser canvas workflow and releases its animation-frame task.
pub(crate) fn stop_web_canvas() {
    // The task consults this slot only between animation-frame awaits. Dropping
    // the viewer here releases decoded study state, rendered frames and canvas
    // listener guards synchronously; task cancellation alone schedules its
    // child future to be dropped on a later executor poll.
    VIEWER.with_borrow_mut(|slot| *slot = None);
    VIEWER_TASK.with_borrow_mut(|slot| slot.take());
    metis_web::metis_stop();
}

/// Returns the number of RITK canvas listener guards retained by the live viewer.
pub(crate) fn web_canvas_listener_count() -> usize {
    VIEWER.with_borrow(|slot| {
        slot.as_ref()
            .map_or(0, |viewer| viewer.surface.listener_count())
    })
}

/// Selects an exact zero-based slice and invalidates cached browser frames.
pub(crate) fn select_web_slice(axis: f64, index: f64) -> Result<(), BrowserSliceSelectionError> {
    let (axis, index) = parse_browser_slice_request(axis, index)?;
    VIEWER.with(|slot| {
        let mut slot = slot
            .try_borrow_mut()
            .map_err(|_| BrowserSliceSelectionError::ViewerBusy)?;
        let viewer = slot
            .as_mut()
            .ok_or(BrowserSliceSelectionError::ViewerNotMounted)?;
        if viewer.app.select_browser_slice(axis, index)? {
            viewer.surface.clear();
        }
        Ok(())
    })
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
