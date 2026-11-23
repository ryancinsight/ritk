//! Grid cursor gesture ownership and patient-to-screen projection.
use super::{
    layout::{screen_coordinate, MAX_GRID_PANELS},
    NativeViewerError, NativeViewerSession,
};
use crate::{
    app::viewer_viewport::ViewerViewport,
    geometry::patient_stack::CursorProjection,
    presentation::{PointerButton, PresentationEvent as Event, ViewportPoint},
    tools::kind::ToolKind,
};
use anyhow::Result;
use metis_platform::Color;
use metis_ui_lang::DisplayList;

#[derive(Default)]
pub(super) struct GridCursor {
    owner: Option<(usize, PointerButton)>,
    marks: [Option<CursorProjection>; MAX_GRID_PANELS],
}

impl GridCursor {
    pub(super) fn visible(&self) -> bool {
        self.marks.iter().any(Option::is_some)
    }
}

impl NativeViewerSession {
    pub(super) fn route_grid_cursor(
        &mut self,
        event: &Event,
        panel: usize,
        viewport: Option<ViewerViewport>,
    ) -> Result<bool, NativeViewerError> {
        let owner = self.grid_cursor.owner;
        let (x, y) = match *event {
            Event::PointerDown {
                x,
                y,
                button,
                modifiers,
            } => {
                if owner.is_some_and(|(_, b)| b == button) {
                    self.grid_cursor = GridCursor::default();
                }
                let app = self.app_for_panel_mut(panel)?;
                let chord = button == PointerButton::Left && modifiers.ctrl() && modifiers.shift();
                let bound = app.pointer.bindings.tool_for(button, app.active_tool)
                    == ToolKind::Crosshair
                    && !(button == PointerButton::Left && (modifiers.ctrl() || modifiers.shift()));
                if !(chord || bound) {
                    return Ok(false);
                }
                if app.cine.enabled || app.patient_stack.is_none() {
                    return Ok(true);
                }
                if self.grid_cursor.owner.is_some() {
                    return Ok(true);
                }
                self.grid_cursor.owner = Some((panel, button));
                (x, y)
            }
            Event::PointerMove { x, y } if owner.is_some_and(|(p, _)| p == panel) => (x, y),
            Event::PointerUp { button, .. } | Event::PointerCancel { button, .. }
                if owner.is_some_and(|(_, b)| b == button) =>
            {
                self.grid_cursor = GridCursor::default();
                return Ok(true);
            }
            Event::FocusLost
            | Event::PointerCaptureLost
            | Event::Destroyed
            | Event::CloseRequested
            | Event::PointerWheel { .. } => {
                self.grid_cursor = GridCursor::default();
                return Ok(false);
            }
            _ => return Ok(false),
        };
        self.grid_cursor.marks.fill(None);
        if !self.viewports.get(panel).is_some_and(|v| v.contains(x, y)) {
            return Ok(true);
        }
        let Some(pixel) = viewport.and_then(|v| v.map(ViewportPoint::new(x, y))) else {
            return Ok(true);
        };
        let source = self.app_for_panel_mut(panel)?;
        let index = source.axis_slice_info(0).0;
        let pixel = [f64::from(pixel.x()), f64::from(pixel.y())];
        let Some(point) = source
            .patient_stack
            .as_ref()
            .and_then(|s| s.patient_point(index, pixel))
        else {
            return Ok(true);
        };
        let mut marks = [None; MAX_GRID_PANELS];
        marks[panel] = Some(CursorProjection {
            index,
            pixel,
            distant: false,
        });
        let count = self.viewports.len();
        // Compute projections before changing slices so the source is stable in any pane.
        let source = if panel == 0 {
            &self.app
        } else {
            &self.compare_panels[panel - 1].app
        };
        if let Some(stack) = &source.patient_stack {
            for (i, target) in std::iter::once(&self.app)
                .chain(self.compare_panels.iter().map(|p| &p.app))
                .take(count)
                .enumerate()
            {
                if i != panel {
                    marks[i] = target
                        .patient_stack
                        .as_ref()
                        .and_then(|t| stack.project_cursor(point, t));
                }
            }
        }
        let apps = std::iter::once(&mut self.app)
            .chain(self.compare_panels.iter_mut().map(|p| &mut p.app))
            .take(count);
        for (i, app) in apps.enumerate() {
            if i != panel
                && let Some(mark) = marks[i]
            {
                app.set_slice_for_axis(0, mark.index);
            }
        }
        self.grid_cursor.marks = marks;
        Ok(true)
    }

    pub(super) fn render_grid_cursor(&mut self) -> Result<()> {
        let mut overlay = DisplayList::default();
        for (viewport, mark) in self.viewports.iter().zip(self.grid_cursor.marks) {
            let Some(mark) = mark else {
                continue;
            };
            let [x, y] = viewport.mapping().project_coordinates(mark.pixel);
            let im = viewport.image;
            let left = im.x.max(f64::from(viewport.panel_x)).ceil();
            let top = im.y.max(f64::from(viewport.panel_y)).ceil();
            let right = (im.x + im.width)
                .min(f64::from(viewport.panel_x + viewport.panel_width))
                .ceil()
                - 1.;
            let bottom = (im.y + im.height)
                .min(f64::from(viewport.panel_y + viewport.panel_height))
                .ceil()
                - 1.;
            if left > right || top > bottom {
                continue;
            }
            let pixel = |v| screen_coordinate(v, "comparison cursor");
            let color = if mark.distant {
                Color::rgba(255, 0, 0, 255)
            } else {
                super::layout::CROSSHAIR_COLOR
            };
            if (left..=right).contains(&x.round()) {
                overlay.append_line(
                    (pixel(x)?, pixel(top)?),
                    (pixel(x)?, pixel(bottom)?),
                    color,
                )?;
            }
            if (top..=bottom).contains(&y.round()) {
                overlay.append_line(
                    (pixel(left)?, pixel(y)?),
                    (pixel(right)?, pixel(y)?),
                    color,
                )?;
            }
        }
        overlay.render_to(&mut self.framebuffer);
        Ok(())
    }
}
