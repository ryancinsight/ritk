//! RITK-owned controls rendered in the visible Métis native client area.

mod events;
mod layout;
mod multi_series;
mod series;
use self::layout::{ChromeGeometry, ChromeLayout};
use self::multi_series::MultiSeriesDialog;
use super::layout::{ViewportArea, WorkspaceLayout};
use super::series_browser::SeriesBrowser;
use crate::app::SnapApp;
use crate::presentation::PresentationFrame;
use crate::tools::kind::ToolKind;
use anyhow::Result;
use metis_platform::Framebuffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Menu {
    File,
    View,
    Tools,
    Window,
    GridPicker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WindowAction {
    OpenMenu(Menu),
    OpenStudy,
    OpenSeriesPicker,
    LoadSelectedSeries,
    Exit,
    SelectSeries(usize),
    OpenSeriesInNextPanel(usize),
    MaximizePanel(usize),
    ClosePanel {
        index: usize,
        kind: PanelCloseKind,
    },
    ToggleActivePanel,
    CloseActivePanel,
    CloseAllPanels,
    ActivateNextPanel,
    ActivatePreviousPanel,
    AssignSeries {
        series_index: usize,
        panel_index: usize,
    },
    SelectTool(ToolKind),
    ToggleSeriesPreview,
    SetLayout(WorkspaceLayout),
    ToggleCrosshair,
    ToggleCine,
    ResetView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PanelCloseKind {
    Close,
    Clear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PointerOwner {
    None,
    Chrome,
    Pane,
    Series(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WindowChromeEvent {
    pub(super) consumed: bool,
    pub(super) repaint: bool,
    pub(super) action: Option<WindowAction>,
}

impl WindowChromeEvent {
    const fn consumed(repaint: bool) -> Self {
        Self {
            consumed: true,
            repaint,
            action: None,
        }
    }

    const fn passed() -> Self {
        Self {
            consumed: false,
            repaint: false,
            action: None,
        }
    }
}

#[derive(Clone)]
pub(super) struct WindowChrome {
    visible: bool,
    open_menu: Option<Menu>,
    multi_series_dialog: Option<MultiSeriesDialog>,
    show_series_preview: bool,
    control_down: bool,
    pointer_owner: PointerOwner,
    pressed_buttons: u8,
}

impl WindowChrome {
    pub(super) const fn new(visible: bool) -> Self {
        Self {
            visible,
            open_menu: None,
            multi_series_dialog: None,
            show_series_preview: true,
            control_down: false,
            pointer_owner: PointerOwner::None,
            pressed_buttons: 0,
        }
    }

    #[cfg(test)]
    pub(super) const fn multi_series_dialog_is_open(&self) -> bool {
        self.multi_series_dialog.is_some()
    }

    #[cfg(test)]
    pub(super) const fn open_menu(&self) -> Option<Menu> {
        self.open_menu
    }

    #[cfg(test)]
    pub(super) fn control_center(
        &self,
        width: u32,
        height: u32,
        app: &SnapApp,
        workspace_layout: WorkspaceLayout,
        open_menu: Option<Menu>,
        action: WindowAction,
    ) -> Result<Option<(i32, i32)>> {
        if !self.visible {
            return Ok(None);
        }
        Ok(ChromeLayout::new(
            width,
            height,
            open_menu,
            app,
            self.show_series_preview,
            workspace_layout,
        )?
        .action_center(action))
    }

    #[cfg(test)]
    pub(super) fn series_card_center(
        &self,
        width: u32,
        height: u32,
        app: &SnapApp,
        workspace_layout: WorkspaceLayout,
        browser: &SeriesBrowser,
        index: usize,
    ) -> Result<Option<(i32, i32)>> {
        if !self.visible {
            return Ok(None);
        }
        Ok(ChromeLayout::new(
            width,
            height,
            self.open_menu,
            app,
            self.show_series_preview,
            workspace_layout,
        )?
        .series_card_center(browser, index))
    }

    pub(super) fn viewport_area(&self, width: u32, height: u32) -> Result<ViewportArea> {
        if !self.visible {
            return Ok(ViewportArea::full(width, height));
        }
        Ok(ChromeGeometry::new(width, height, self.show_series_preview)?.viewport_area)
    }

    pub(super) fn render(
        &self,
        framebuffer: &mut Framebuffer,
        app: &SnapApp,
        series_previews: &[Option<&PresentationFrame>],
        mut browser: Option<&mut SeriesBrowser>,
        workspace_layout: WorkspaceLayout,
        active_panel: usize,
        displayed_series: &[Option<usize>],
    ) -> Result<()> {
        if !self.visible {
            return Ok(());
        }
        let layout = ChromeLayout::new(
            framebuffer.width(),
            framebuffer.height(),
            self.open_menu,
            app,
            self.show_series_preview,
            workspace_layout,
        )?;
        if let Some(series_browser) = browser.as_deref_mut() {
            series::prepare_thumbnails(
                series_browser,
                layout.series_preview_area(),
                displayed_series,
            )?;
        }
        layout.render(
            framebuffer,
            app,
            series_previews,
            browser.as_deref(),
            workspace_layout,
            active_panel,
            displayed_series,
        )?;
        if let (Some(dialog), Some(browser)) = (&self.multi_series_dialog, browser.as_deref()) {
            dialog.render(framebuffer, browser)?;
        }
        Ok(())
    }

    pub(super) fn toggle_series_preview(&mut self) {
        self.show_series_preview = !self.show_series_preview;
        self.open_menu = None;
    }

    pub(super) fn open_series_picker(&mut self, browser: Option<&SeriesBrowser>) -> Result<bool> {
        let Some(browser) = browser else {
            return Ok(false);
        };
        self.multi_series_dialog = Some(MultiSeriesDialog::new(browser)?);
        self.open_menu = None;
        Ok(true)
    }

    pub(super) fn take_selected_series(
        &mut self,
    ) -> Option<arrayvec::ArrayVec<usize, { super::layout::MAX_GRID_PANELS }>> {
        self.multi_series_dialog
            .take()
            .map(|dialog| dialog.selected().clone())
    }

    pub(super) fn cancel_pointer_capture(&mut self) {
        self.pointer_owner = PointerOwner::None;
        self.pressed_buttons = 0;
    }
}

#[cfg(test)]
#[path = "window_controls/tests.rs"]
mod tests;
