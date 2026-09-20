//! Browser surface lifecycle and canvas event routing.

use super::browser_canvas::BrowserCanvas;
use super::browser_geometry::{viewport_for_display, PhysicalCanvasAspect};
use super::browser_semantics::BrowserCanvasSemantics;
use super::SnapApp;
use crate::app::action_adapter::ViewerActionDisposition;
use crate::presentation::PresentationFrame;
use crate::render::{FrameRenderScratch, ProjectionStatistic};

pub(super) enum BrowserSurface {
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
    OrthogonalWithProjection {
        canvases: Box<[BrowserCanvas; 4]>,
        frames: Option<[PresentationFrame; 4]>,
        projection_pixels: Vec<f32>,
        statistic: ProjectionStatistic,
        scratch: FrameRenderScratch,
        dirty: bool,
    },
}

impl BrowserSurface {
    pub(super) fn single(canvas: BrowserCanvas) -> Self {
        Self::Single {
            canvas,
            frame: None,
            scratch: FrameRenderScratch::default(),
            dirty: true,
        }
    }

    pub(super) fn orthogonal(canvases: [BrowserCanvas; 3]) -> Self {
        Self::Orthogonal {
            canvases: Box::new(canvases),
            frames: None,
            scratch: FrameRenderScratch::default(),
            dirty: true,
        }
    }

    pub(super) fn orthogonal_with_projection(
        canvases: [BrowserCanvas; 4],
        statistic: ProjectionStatistic,
    ) -> Self {
        Self::OrthogonalWithProjection {
            canvases: Box::new(canvases),
            frames: None,
            projection_pixels: Vec::new(),
            statistic,
            scratch: FrameRenderScratch::default(),
            dirty: true,
        }
    }
    pub(super) fn listener_count(&self) -> usize {
        match self {
            Self::Single { canvas, .. } => canvas.listener_count(),
            Self::Orthogonal { canvases, .. } => {
                canvases.iter().map(BrowserCanvas::listener_count).sum()
            }
            Self::OrthogonalWithProjection { canvases, .. } => canvases
                .iter()
                .take(3)
                .map(BrowserCanvas::listener_count)
                .sum(),
        }
    }

    pub(super) fn clear(&mut self) {
        match self {
            Self::Single { dirty, .. }
            | Self::Orthogonal { dirty, .. }
            | Self::OrthogonalWithProjection { dirty, .. } => *dirty = true,
        }
    }

    pub(super) fn render_and_present(&mut self, app: &SnapApp) -> std::io::Result<()> {
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
            Self::OrthogonalWithProjection {
                canvases,
                frames,
                projection_pixels,
                statistic,
                scratch,
                dirty,
            } => {
                if app.loaded.is_none() {
                    if let Some(frames) = frames.take() {
                        for mut frame in frames {
                            frame.reclaim_storage(scratch);
                        }
                    }
                    projection_pixels.clear();
                    *dirty = false;
                    return Ok(());
                }
                let rendered = *dirty || frames.is_none();
                if frames.is_none() {
                    *frames = Some([
                        PresentationFrame::empty(),
                        PresentationFrame::empty(),
                        PresentationFrame::empty(),
                        PresentationFrame::empty(),
                    ]);
                }
                if rendered {
                    let frames = frames.as_mut().ok_or_else(|| {
                        std::io::Error::other("browser projection frames were not initialized")
                    })?;
                    let (orthogonal, projection) = frames.split_at_mut(3);
                    app.render_browser_frames_into(orthogonal, scratch)
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                    let projection = projection.first_mut().ok_or_else(|| {
                        std::io::Error::other("browser projection frame was not initialized")
                    })?;
                    app.render_browser_projection_into(
                        projection,
                        *statistic,
                        projection_pixels,
                        scratch,
                    )
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

    pub(super) fn publish_semantics(&mut self, app: &SnapApp) -> std::io::Result<()> {
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
            Self::OrthogonalWithProjection {
                canvases,
                frames,
                statistic,
                ..
            } => {
                let (orthogonal, projection) = canvases.split_at_mut(3);
                for (axis, canvas) in orthogonal.iter_mut().enumerate() {
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
                let projection = projection.first_mut().ok_or_else(|| {
                    std::io::Error::other("browser projection canvas was not initialized")
                })?;
                let frame = frames.as_ref().and_then(|frames| frames.get(3));
                projection.publish_projection(
                    app.loaded.is_some(),
                    frame,
                    *statistic,
                    physical_aspect(frame)?,
                )
            }
        }
    }

    pub(super) fn apply_events(
        &mut self,
        app: &mut SnapApp,
    ) -> std::io::Result<ViewerActionDisposition> {
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
            Self::OrthogonalWithProjection {
                canvases, frames, ..
            } => {
                let mut repaint = false;
                for (axis, canvas) in canvases.iter_mut().take(3).enumerate() {
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
