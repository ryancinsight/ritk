//! Responsive browser canvas composition for the RITK presentation surface.

use super::browser_canvas::BrowserCanvas;
use super::browser_semantics::BrowserCanvasSemantics;
use super::web_surface::{apply_canvas_events, physical_aspect};
use super::SnapApp;
use crate::app::action_adapter::ViewerActionDisposition;
use crate::presentation::{PaneLayout, PaneRole, PresentationFrame};
use crate::render::{FrameRenderScratch, ProjectionStatistic};
use moirai_pal::wasm::WebElement;
use std::io;

const PANE_GAP: u32 = 4;

struct ResponsiveCanvas {
    id: String,
    role: PaneRole,
    canvas: BrowserCanvas,
    input_enabled: bool,
}

impl ResponsiveCanvas {
    fn new(id: &str, role: PaneRole, layout: PaneLayout, index: usize) -> io::Result<Self> {
        let input_enabled = layout.roles().contains(&role) && matches!(role, PaneRole::Axis(_));
        let canvas = if input_enabled {
            BrowserCanvas::from_id(id)?
        } else {
            BrowserCanvas::from_id_without_input(id)?
        };
        canvas.set_pane_metadata(layout, role, layout.roles().contains(&role), index)?;
        Ok(Self {
            id: id.to_owned(),
            role,
            canvas,
            input_enabled,
        })
    }

    fn set_layout(&mut self, layout: PaneLayout, index: usize) -> io::Result<()> {
        let visible = layout.roles().contains(&self.role);
        let input_enabled = visible && matches!(self.role, PaneRole::Axis(_));
        if input_enabled != self.input_enabled {
            let canvas = if input_enabled {
                BrowserCanvas::from_id(&self.id)?
            } else {
                BrowserCanvas::from_id_without_input(&self.id)?
            };
            self.canvas = canvas;
            self.input_enabled = input_enabled;
        }
        self.canvas
            .set_pane_metadata(layout, self.role, visible, index)?;
        Ok(())
    }

    fn listener_count(&self) -> usize {
        self.canvas.listener_count()
    }
}

pub(super) struct ResponsiveSurface {
    container: WebElement,
    canvases: Box<[ResponsiveCanvas; 4]>,
    frames: Option<[PresentationFrame; 4]>,
    projection_pixels: Vec<f32>,
    statistic: ProjectionStatistic,
    scratch: FrameRenderScratch,
    layout: PaneLayout,
    dirty: bool,
}

impl ResponsiveSurface {
    pub(super) fn new(
        container: WebElement,
        canvas_ids: [String; 4],
        statistic: ProjectionStatistic,
    ) -> io::Result<Self> {
        let layout = layout_for(&container);
        let [axial_id, coronal_id, sagittal_id, projection_id] = canvas_ids;
        let canvases = [
            ResponsiveCanvas::new(&axial_id, PaneRole::Axis(0), layout, 0)?,
            ResponsiveCanvas::new(&coronal_id, PaneRole::Axis(1), layout, 1)?,
            ResponsiveCanvas::new(&sagittal_id, PaneRole::Axis(2), layout, 2)?,
            ResponsiveCanvas::new(&projection_id, PaneRole::Projection, layout, 3)?,
        ];
        let surface = Self {
            container,
            canvases: Box::new(canvases),
            frames: None,
            projection_pixels: Vec::new(),
            statistic,
            scratch: FrameRenderScratch::default(),
            layout,
            dirty: true,
        };
        surface.set_container_metadata(layout)?;
        Ok(surface)
    }

    pub(super) fn refresh_layout(&mut self) -> io::Result<bool> {
        let next = layout_for(&self.container);
        if next == self.layout {
            return Ok(false);
        }
        for (index, canvas) in self.canvases.iter_mut().enumerate() {
            canvas.set_layout(next, index)?;
        }
        self.set_container_metadata(next)?;
        self.layout = next;
        self.dirty = true;
        Ok(true)
    }

    pub(super) fn clear(&mut self) {
        self.dirty = true;
    }

    pub(super) fn listener_count(&self) -> usize {
        self.canvases
            .iter()
            .map(ResponsiveCanvas::listener_count)
            .sum()
    }

    pub(super) fn render_and_present(&mut self, app: &SnapApp) -> io::Result<()> {
        if app.loaded.is_none() {
            if let Some(frames) = self.frames.take() {
                for mut frame in frames {
                    frame.reclaim_storage(&mut self.scratch);
                }
            }
            self.projection_pixels.clear();
            self.dirty = false;
            return Ok(());
        }
        let rendered = self.dirty || self.frames.is_none();
        if self.frames.is_none() {
            self.frames = Some([
                PresentationFrame::empty(),
                PresentationFrame::empty(),
                PresentationFrame::empty(),
                PresentationFrame::empty(),
            ]);
        }
        if rendered {
            let frames = self.frames.as_mut().ok_or_else(|| {
                io::Error::other("responsive browser frames were not initialized")
            })?;
            let (orthogonal, projection) = frames.split_at_mut(3);
            app.render_browser_frames_into(orthogonal, &mut self.scratch)
                .map_err(|error| io::Error::other(error.to_string()))?;
            let projection = projection.first_mut().ok_or_else(|| {
                io::Error::other("responsive browser projection frame was not initialized")
            })?;
            app.render_browser_projection_into(
                projection,
                self.statistic,
                &mut self.projection_pixels,
                &mut self.scratch,
            )
            .map_err(|error| io::Error::other(error.to_string()))?;
            self.dirty = false;
        }
        if let Some(frames) = self.frames.as_ref() {
            for (canvas, frame) in self.canvases.iter_mut().zip(frames) {
                if canvas_visible(canvas.role, self.layout) && rendered {
                    canvas.canvas.present_rendered_frame(frame)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn publish_semantics(&mut self, app: &SnapApp) -> io::Result<()> {
        let snapshot = app
            .presentation_snapshot()
            .with_window_preset_index(app.browser_window_preset_index());
        let Some(frames) = self.frames.as_ref() else {
            return Ok(());
        };
        for (axis, canvas) in self.canvases.iter_mut().take(3).enumerate() {
            let frame = frames.get(axis);
            let semantics = BrowserCanvasSemantics::from_snapshot(snapshot.with_axis(axis), frame);
            canvas
                .canvas
                .publish_semantics(semantics, physical_aspect(frame)?)?;
        }
        let projection = self.canvases.get_mut(3).ok_or_else(|| {
            io::Error::other("responsive browser projection canvas was not initialized")
        })?;
        projection.canvas.publish_projection(
            app.loaded.is_some(),
            frames.get(3),
            self.statistic,
            physical_aspect(frames.get(3))?,
        )
    }

    pub(super) fn apply_events(
        &mut self,
        app: &mut SnapApp,
    ) -> io::Result<ViewerActionDisposition> {
        let Some(frames) = self.frames.as_ref() else {
            return Ok(ViewerActionDisposition::Continue { repaint: false });
        };
        let mut repaint = false;
        for (axis, canvas) in self.canvases.iter_mut().take(3).enumerate() {
            if !canvas.input_enabled {
                continue;
            }
            match apply_canvas_events(app, axis, &mut canvas.canvas, frames.get(axis))? {
                ViewerActionDisposition::Continue { repaint: needed } => repaint |= needed,
                ViewerActionDisposition::Exit => return Ok(ViewerActionDisposition::Exit),
            }
        }
        Ok(ViewerActionDisposition::Continue { repaint })
    }

    fn set_container_metadata(&self, layout: PaneLayout) -> io::Result<()> {
        self.container
            .set_attribute("data-ritk-pane-layout", layout.label())?;
        self.container.set_style_property("display", "grid")?;
        self.container
            .set_style_property("gap", &format!("{PANE_GAP}px"))?;
        let columns = if matches!(layout, PaneLayout::Single) {
            "minmax(0, 1fr)"
        } else {
            "repeat(2, minmax(0, 1fr))"
        };
        let rows = if matches!(layout, PaneLayout::Quad) {
            "repeat(2, minmax(0, 1fr))"
        } else {
            "minmax(0, 1fr)"
        };
        self.container
            .set_style_property("grid-template-columns", columns)?;
        self.container
            .set_style_property("grid-template-rows", rows)
    }
}

fn canvas_visible(role: PaneRole, layout: PaneLayout) -> bool {
    layout.roles().contains(&role)
}

fn layout_for(container: &WebElement) -> PaneLayout {
    let size = container.bounding_size();
    PaneLayout::responsive(css_dimension(size.width()), css_dimension(size.height()))
}

fn css_dimension(value: f64) -> u32 {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    let bounded = value.min(f64::from(u32::MAX)).round();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the finite CSS extent is bounded by u32 before conversion"
    )]
    {
        bounded as u32
    }
}
