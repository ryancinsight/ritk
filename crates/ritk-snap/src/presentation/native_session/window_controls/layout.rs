//! RadiAnt-clone native menus, grouped tools, series preview and status bar.

#[cfg(test)]
use super::super::layout::ViewportArea;
mod construction;
mod geometry;
mod render;
use super::super::series_browser::SeriesBrowser;
use super::series;
use super::WindowAction;
use crate::app::SnapApp;
use crate::presentation::PresentationFrame;
use anyhow::Result;
use arrayvec::ArrayVec;
pub(super) use geometry::ChromeGeometry;
use metis_platform::{Framebuffer, Rect};

const CONTROL_CAPACITY: usize = 40;

#[derive(Clone, Copy)]
enum ControlKind {
    MenuTab,
    Toolbar,
    MenuItem,
    Grid,
}

#[derive(Clone, Copy)]
struct ChromeControl {
    rect: Rect,
    label: &'static str,
    action: WindowAction,
    kind: ControlKind,
    active: bool,
}

#[derive(Clone, Copy)]
struct ToolbarGroup {
    x: i32,
    label: &'static str,
}

pub(super) struct ChromeLayout {
    geometry: ChromeGeometry,
    controls: ArrayVec<ChromeControl, CONTROL_CAPACITY>,
    groups: ArrayVec<ToolbarGroup, 8>,
    grid_popup: Option<Rect>,
}

impl ChromeLayout {
    pub(super) fn render(
        &self,
        framebuffer: &mut Framebuffer,
        app: &SnapApp,
        series_previews: &[Option<&PresentationFrame>],
        browser: Option<&SeriesBrowser>,
        workspace_layout: super::super::layout::WorkspaceLayout,
        active_panel: usize,
        displayed_series: &[Option<usize>],
    ) -> Result<()> {
        render::render(
            self,
            framebuffer,
            app,
            series_previews,
            browser,
            workspace_layout,
            active_panel,
            displayed_series,
        )
    }

    pub(super) fn action_at(
        &self,
        x: f64,
        y: f64,
        browser: Option<&SeriesBrowser>,
    ) -> Option<WindowAction> {
        self.controls
            .iter()
            .rev()
            .find(|control| rect_contains(control.rect, x, y))
            .map(|control| control.action)
            .or_else(|| {
                let browser = browser?;
                let index =
                    series::navigator::index_at(browser, self.geometry.series_preview, x, y)?;
                Some(WindowAction::SelectSeries(index))
            })
    }

    #[cfg(test)]
    pub(super) fn action_center(&self, action: WindowAction) -> Option<(i32, i32)> {
        let control = self
            .controls
            .iter()
            .find(|control| control.action == action)?;
        Some((
            control.rect.x.checked_add(control.rect.width / 2)?,
            control.rect.y.checked_add(control.rect.height / 2)?,
        ))
    }

    pub(super) fn series_index_at(
        &self,
        browser: Option<&SeriesBrowser>,
        x: f64,
        y: f64,
    ) -> Option<usize> {
        let browser = browser?;
        series::navigator::index_at(browser, self.geometry.series_preview, x, y)
    }

    #[cfg(test)]
    pub(super) fn series_card_center(
        &self,
        browser: &SeriesBrowser,
        index: usize,
    ) -> Option<(i32, i32)> {
        series::navigator::card_center(browser, self.geometry.series_preview, index)
    }

    #[cfg(test)]
    pub(super) const fn viewport_area(&self) -> ViewportArea {
        self.geometry.viewport_area
    }

    pub(super) const fn series_preview_area(&self) -> Rect {
        self.geometry.series_preview
    }

    pub(super) fn visible_series(&self) -> usize {
        series::navigator::visible_count(self.geometry.series_preview)
    }

    pub(super) fn series_scroll_direction_at(
        &self,
        browser: &SeriesBrowser,
        x: f64,
        y: f64,
    ) -> anyhow::Result<Option<i32>> {
        series::scrollbar::page_direction(self.geometry.series_preview, browser, x, y)
    }

    pub(super) fn series_contains(&self, x: f64, y: f64) -> bool {
        rect_contains(self.geometry.series_preview, x, y)
    }

    pub(super) fn popup_contains(&self, x: f64, y: f64) -> bool {
        self.grid_popup
            .is_some_and(|popup| rect_contains(popup, x, y))
            || self.controls.iter().any(|control| {
                matches!(control.kind, ControlKind::MenuItem) && rect_contains(control.rect, x, y)
            })
    }

    pub(super) fn owns_pointer(&self, x: f64, y: f64) -> bool {
        rect_contains(self.geometry.menu_bar, x, y)
            || rect_contains(self.geometry.toolbar, x, y)
            || self.series_contains(x, y)
            || rect_contains(self.geometry.status_bar, x, y)
            || self.controls.iter().any(|control| {
                matches!(control.kind, ControlKind::MenuItem | ControlKind::Grid)
                    && rect_contains(control.rect, x, y)
            })
    }

    #[cfg(test)]
    pub(super) fn controls_fit_within(&self, width: u32, height: u32) -> bool {
        let width = i64::from(width);
        let height = i64::from(height);
        self.controls.iter().all(|control| {
            let left = i64::from(control.rect.x);
            let top = i64::from(control.rect.y);
            let right = left + i64::from(control.rect.width);
            let bottom = top + i64::from(control.rect.height);
            left >= 0 && top >= 0 && right <= width && bottom <= height
        })
    }
}

fn rect_contains(rect: Rect, x: f64, y: f64) -> bool {
    if rect.width <= 0 || rect.height <= 0 || !x.is_finite() || !y.is_finite() {
        return false;
    }
    let left = f64::from(rect.x);
    let top = f64::from(rect.y);
    x >= left && y >= top && x < left + f64::from(rect.width) && y < top + f64::from(rect.height)
}
