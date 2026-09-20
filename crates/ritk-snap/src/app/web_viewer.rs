//! Métis browser canvas viewer workflow.
//!
//! This module is the first browser migration slice beyond the eframe shell:
//! RITK consumes the bounded byte handoff from Métis, applies its existing
//! dropped-input classifier and presents the selected RITK frame through the
//! borrowed canvas seam. It also publishes a bounded semantic snapshot on
//! each canvas for consumer-owned workflow assertions. DICOM parsing and
//! viewer state stay in [`SnapApp`].

use super::browser_canvas::BrowserCanvas;
use super::browser_cine::{parse_browser_cine_rate_request, BrowserCineControlError};
use super::browser_geometry::{viewport_for_display, PhysicalCanvasAspect};
use super::browser_semantics::BrowserCanvasSemantics;
use super::browser_slice_selection::{parse_browser_slice_request, BrowserSliceSelectionError};
use super::browser_tool::{parse_browser_tool_request, BrowserToolError};
use super::SnapApp;
use crate::app::action_adapter::ViewerActionDisposition;
use crate::presentation::PresentationFrame;
use crate::render::FrameRenderScratch;
use crate::ui::decide_dropped_input_action;
use moirai_pal::wasm::{spawn_local_with_handle, LocalTaskHandle, WebAnimationFrame};
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

enum BrowserSurface {
    Single {
        canvas: BrowserCanvas,
        frame: Option<PresentationFrame>,
        scratch: FrameRenderScratch,
        dirty: bool,
    },
    Orthogonal {
        canvases: Box<[BrowserCanvas; 3]>,
        frames: Option<[PresentationFrame; 3]>,
        scratch: FrameRenderScratch,
        dirty: bool,
    },
}

impl BrowserSurface {
    fn listener_count(&self) -> usize {
        match self {
            Self::Single { canvas, .. } => canvas.listener_count(),
            Self::Orthogonal { canvases, .. } => {
                canvases.iter().map(BrowserCanvas::listener_count).sum()
            }
        }
    }

    fn clear(&mut self) {
        match self {
            Self::Single { dirty, .. } | Self::Orthogonal { dirty, .. } => *dirty = true,
        }
    }

    fn render_and_present(&mut self, app: &SnapApp) -> std::io::Result<()> {
        match self {
            Self::Single {
                canvas,
                frame,
                scratch,
                dirty,
            } => {
                if app.loaded.is_none() {
                    if let Some(mut frame) = frame.take() {
                        frame.reclaim_storage(scratch);
                    }
                    *dirty = false;
                    return Ok(());
                }
                let rendered = *dirty || frame.is_none();
                if frame.is_none() {
                    *frame = Some(PresentationFrame::empty());
                }
                if rendered {
                    let frame = frame.as_mut().ok_or_else(|| {
                        std::io::Error::other("browser frame was not initialized")
                    })?;
                    app.render_browser_frame_into(frame, scratch)
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                    *dirty = false;
                }
                if let Some(frame) = frame.as_ref() {
                    if rendered {
                        canvas.present_rendered_frame(frame)?;
                    }
                }
            }
            Self::Orthogonal {
                canvases,
                frames,
                scratch,
                dirty,
            } => {
                if app.loaded.is_none() {
                    if let Some(frames) = frames.take() {
                        for mut frame in frames {
                            frame.reclaim_storage(scratch);
                        }
                    }
                    *dirty = false;
                    return Ok(());
                }
                let rendered = *dirty || frames.is_none();
                if frames.is_none() {
                    *frames = Some([
                        PresentationFrame::empty(),
                        PresentationFrame::empty(),
                        PresentationFrame::empty(),
                    ]);
                }
                if rendered {
                    let frames = frames.as_mut().ok_or_else(|| {
                        std::io::Error::other("browser frames were not initialized")
                    })?;
                    app.render_browser_frames_into(frames, scratch)
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                    *dirty = false;
                }
                if let Some(frames) = frames.as_ref() {
                    for (canvas, frame) in canvases.iter_mut().zip(frames) {
                        if rendered {
                            canvas.present_rendered_frame(frame)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn publish_semantics(&mut self, app: &SnapApp) -> std::io::Result<()> {
        match self {
            Self::Single { canvas, frame, .. } => {
                let axis = app.axis;
                let (slice_index, slice_count) = app.axis_slice_info(axis);
                let (window_center, window_width) = app.browser_window_level_values();
                let semantics = BrowserCanvasSemantics::from_state(
                    app.loaded.is_some(),
                    axis,
                    slice_index,
                    slice_count,
                    frame.as_ref(),
                    app.browser_cine_enabled(),
                    app.browser_cine_rate(),
                    window_center,
                    window_width,
                    app.browser_window_preset_index(),
                    app.browser_tool_index(),
                    app.active_tool.label(),
                );
                let physical_aspect = physical_aspect(frame.as_ref())?;
                canvas.publish_semantics(semantics, physical_aspect)
            }
            Self::Orthogonal {
                canvases, frames, ..
            } => {
                for (axis, canvas) in canvases.iter_mut().enumerate() {
                    let frame = frames.as_ref().and_then(|frames| frames.get(axis));
                    let (slice_index, slice_count) = app.axis_slice_info(axis);
                    let (window_center, window_width) = app.browser_window_level_values();
                    let semantics = BrowserCanvasSemantics::from_state(
                        app.loaded.is_some(),
                        axis,
                        slice_index,
                        slice_count,
                        frame,
                        app.browser_cine_enabled(),
                        app.browser_cine_rate(),
                        window_center,
                        window_width,
                        app.browser_window_preset_index(),
                        app.browser_tool_index(),
                        app.active_tool.label(),
                    );
                    let physical_aspect = physical_aspect(frame)?;
                    canvas.publish_semantics(semantics, physical_aspect)?;
                }
                Ok(())
            }
        }
    }

    fn apply_events(&mut self, app: &mut SnapApp) -> std::io::Result<ViewerActionDisposition> {
        match self {
            Self::Single { canvas, frame, .. } => {
                apply_canvas_events(app, 0, canvas, frame.as_ref())
            }
            Self::Orthogonal {
                canvases, frames, ..
            } => {
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
    frame: Option<&PresentationFrame>,
) -> std::io::Result<Option<PhysicalCanvasAspect>> {
    let Some(frame) = frame else {
        return Ok(None);
    };
    PhysicalCanvasAspect::from_display_spacing(
        frame.display_spacing(),
        frame.width(),
        frame.height(),
    )
    .map(Some)
}

fn apply_canvas_events(
    app: &mut SnapApp,
    axis: usize,
    canvas: &mut BrowserCanvas,
    frame: Option<&PresentationFrame>,
) -> std::io::Result<ViewerActionDisposition> {
    let events = match canvas.take_events() {
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
        scratch: FrameRenderScratch::default(),
        dirty: true,
    }))
}

/// Starts the RITK byte-drop workflow with an explicit WebGPU canvas.
pub(crate) async fn start_web_canvas_gpu(canvas_id: String) -> Result<(), JsValue> {
    stop_web_canvas();
    let canvas = match BrowserCanvas::from_id_gpu(&canvas_id).await {
        Ok(canvas) => canvas,
        Err(error) => return Err(JsValue::from_str(&error.to_string())),
    };
    metis_web::metis_start();
    launch_browser_viewer(BrowserViewer::new(BrowserSurface::Single {
        canvas,
        frame: None,
        scratch: FrameRenderScratch::default(),
        dirty: true,
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
        scratch: FrameRenderScratch::default(),
        dirty: true,
    }))
}

/// Starts the RITK byte-drop workflow with three explicit WebGPU canvases.
pub(crate) async fn start_web_orthogonal_canvases_gpu(
    canvas_ids: [String; 3],
) -> Result<(), JsValue> {
    stop_web_canvas();
    let [axial_id, coronal_id, sagittal_id] = canvas_ids;
    let axial = BrowserCanvas::from_id_gpu(&axial_id)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let coronal = BrowserCanvas::from_id_gpu(&coronal_id)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let sagittal = BrowserCanvas::from_id_gpu(&sagittal_id)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let canvases = [axial, coronal, sagittal];
    metis_web::metis_start();
    launch_browser_viewer(BrowserViewer::new(BrowserSurface::Orthogonal {
        canvases: Box::new(canvases),
        frames: None,
        scratch: FrameRenderScratch::default(),
        dirty: true,
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

/// Applies one exact loaded-modality window/level preset and invalidates frames.
pub(crate) fn set_web_window_preset(
    index: f64,
) -> Result<(), super::browser_window_preset::BrowserWindowPresetError> {
    let index = super::browser_window_preset::parse_browser_window_preset_request(index)?;
    VIEWER.with(|slot| {
        let mut slot = slot
            .try_borrow_mut()
            .map_err(|_| super::browser_window_preset::BrowserWindowPresetError::ViewerBusy)?;
        let viewer = slot
            .as_mut()
            .ok_or(super::browser_window_preset::BrowserWindowPresetError::ViewerNotMounted)?;
        if viewer.app.apply_browser_window_preset(index)? {
            viewer.surface.clear();
        }
        Ok(())
    })
}

/// Toggles cine playback for the loaded browser study.
pub(crate) fn toggle_web_cine() -> Result<bool, BrowserCineControlError> {
    VIEWER.with(|slot| {
        let mut slot = slot
            .try_borrow_mut()
            .map_err(|_| BrowserCineControlError::ViewerBusy)?;
        let viewer = slot
            .as_mut()
            .ok_or(BrowserCineControlError::ViewerNotMounted)?;
        viewer.app.toggle_browser_cine()
    })
}

/// Applies one exact bounded cine rate to the loaded browser study.
pub(crate) fn set_web_cine_rate(rate: f64) -> Result<bool, BrowserCineControlError> {
    let rate = parse_browser_cine_rate_request(rate)?;
    VIEWER.with(|slot| {
        let mut slot = slot
            .try_borrow_mut()
            .map_err(|_| BrowserCineControlError::ViewerBusy)?;
        let viewer = slot
            .as_mut()
            .ok_or(BrowserCineControlError::ViewerNotMounted)?;
        viewer.app.set_browser_cine_rate(rate)
    })
}

/// Selects one loaded-study interaction tool from the stable RITK table.
pub(crate) fn select_web_tool(index: f64) -> Result<bool, BrowserToolError> {
    let index = parse_browser_tool_request(index)?;
    VIEWER.with(|slot| {
        let mut slot = slot
            .try_borrow_mut()
            .map_err(|_| BrowserToolError::ViewerBusy)?;
        let viewer = slot.as_mut().ok_or(BrowserToolError::ViewerNotMounted)?;
        viewer.app.select_browser_tool(index)
    })
}

/// Returns the number of interaction tools available to the browser palette.
pub(crate) fn web_tool_count() -> Result<usize, BrowserToolError> {
    VIEWER.with(|slot| {
        let slot = slot
            .try_borrow()
            .map_err(|_| BrowserToolError::ViewerBusy)?;
        slot.as_ref()
            .ok_or(BrowserToolError::ViewerNotMounted)
            .map(|_| SnapApp::browser_tool_count())
    })
}

/// Returns one interaction-tool label for browser palette construction.
pub(crate) fn web_tool_name(index: f64) -> Result<String, BrowserToolError> {
    let index = parse_browser_tool_request(index)?;
    VIEWER.with(|slot| {
        let slot = slot
            .try_borrow()
            .map_err(|_| BrowserToolError::ViewerBusy)?;
        slot.as_ref().ok_or(BrowserToolError::ViewerNotMounted)?;
        SnapApp::browser_tool_name(index).map(str::to_owned)
    })
}

/// Returns the loaded-modality window/level preset count.
pub(crate) fn web_window_preset_count(
) -> Result<usize, super::browser_window_preset::BrowserWindowPresetError> {
    VIEWER.with(|slot| {
        let slot = slot
            .try_borrow()
            .map_err(|_| super::browser_window_preset::BrowserWindowPresetError::ViewerBusy)?;
        let viewer = slot
            .as_ref()
            .ok_or(super::browser_window_preset::BrowserWindowPresetError::ViewerNotMounted)?;
        viewer.app.browser_window_preset_count()
    })
}

/// Returns one loaded-modality window/level preset name.
pub(crate) fn web_window_preset_name(
    index: f64,
) -> Result<String, super::browser_window_preset::BrowserWindowPresetError> {
    let index = super::browser_window_preset::parse_browser_window_preset_request(index)?;
    VIEWER.with(|slot| {
        let slot = slot
            .try_borrow()
            .map_err(|_| super::browser_window_preset::BrowserWindowPresetError::ViewerBusy)?;
        let viewer = slot
            .as_ref()
            .ok_or(super::browser_window_preset::BrowserWindowPresetError::ViewerNotMounted)?;
        viewer
            .app
            .browser_window_preset_name(index)
            .map(str::to_owned)
    })
}

impl SnapApp {
    fn render_browser_frame_into(
        &self,
        frame: &mut PresentationFrame,
        scratch: &mut FrameRenderScratch,
    ) -> anyhow::Result<()> {
        let Some(volume) = self.loaded.as_ref() else {
            return Ok(());
        };
        let window_level = self.browser_window_level();
        let slice_index = match self.axis {
            0 => self.viewer_state.slice_index,
            1 => self.coronal_slice,
            _ => self.sagittal_slice,
        };
        frame.render_slice_into(
            volume,
            self.axis,
            slice_index,
            window_level,
            self.colormap,
            scratch,
        )
    }

    fn render_browser_frames_into(
        &self,
        frames: &mut [PresentationFrame; 3],
        scratch: &mut FrameRenderScratch,
    ) -> anyhow::Result<()> {
        let Some(volume) = self.loaded.as_ref() else {
            return Ok(());
        };
        let indices = [0_usize, 1, 2].map(|axis| self.axis_slice_info(axis).0);
        for (axis, frame) in frames.iter_mut().enumerate() {
            frame.render_slice_into(
                volume,
                axis,
                indices[axis],
                self.browser_window_level(),
                self.colormap,
                scratch,
            )?;
        }
        Ok(())
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
