//! RITK-owned controls rendered in the visible Métis native client area.

mod calibration;
mod events;
mod key_bindings;
mod layout;
mod multi_series;
mod series;
mod slice_drag;
use self::layout::{ChromeGeometry, ChromeLayout};
use self::multi_series::MultiSeriesDialog;
use super::deidentify::{AnonymizationScope, Deidentification};
use super::layout::{ViewportArea, WorkspaceLayout};
use super::pressed_buttons::PressedButtons;
use super::series_browser::SeriesBrowser;
use crate::app::SnapApp;
use crate::presentation::{
    PointerButton, PresentationEvent, PresentationFrame, PresentationModifiers,
};
use crate::tools::kind::ToolKind;
use anyhow::Result;
use metis_platform::Framebuffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Menu {
    File,
    View,
    Tools,
    Window,
    Display,
    Anonymize,
    GridPicker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WindowAction {
    OpenMenu(Menu),
    OpenStudy,
    Anonymize(AnonymizationScope, Deidentification),
    RepairStudy,
    OpenSeriesPicker,
    LoadSelectedSeries,
    Exit,
    SelectSeries(usize),
    SelectSlice {
        panel_index: usize,
        axis: usize,
        index: Option<usize>,
    },
    BrowseSeries {
        series_index: usize,
        panel_index: usize,
    },
    OpenSeriesInNextPanel(usize),
    MaximizePanel(usize),
    ClosePanel {
        index: usize,
        kind: PanelCloseKind,
    },
    ToggleActivePanel,
    CloseActivePanel,
    CloseAllPanels,
    CancelInteraction,
    ActivateNextPanel,
    ActivatePreviousPanel,
    AssignSeries {
        series_index: usize,
        panel_index: usize,
    },
    SelectTool(ToolKind),
    LengthCalibration,
    ApplyLengthCalibration {
        panel: usize,
        reference: crate::app::LengthCalibrationReference,
        known_mm: Option<u32>,
    },
    AssignPointerTool {
        button: PointerButton,
        tool: ToolKind,
    },
    ToggleSeriesPreview,
    SetLayout(WorkspaceLayout),
    ToggleCrosshair,
    ToggleOverlay,
    ToggleAnonymizedOverlay,
    ToggleCine,
    StepEnhancedPetVolume(isize),
    ResetView,
    ApplyWindowPreset {
        index: usize,
    },
    ToggleInvert,
    SetColormap(crate::render::NamedColorMap),
    RotateCw,
    RotateCcw,
    RotateHalfTurn,
    FlipHorizontal,
    FlipVertical,
    ResetOrientation,
    ZoomFit,
    ZoomOneToOne,
    ApplyCustomPreset {
        index: usize,
    },
    SaveCustomPreset {
        index: Option<usize>,
    },
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
    Slice(slice_drag::SliceDrag),
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
    calibration: Option<calibration::CalibrationDialog>,
    show_series_preview: bool,
    modifiers: PresentationModifiers,
    pointer_owner: PointerOwner,
    pressed_buttons: PressedButtons,
}

impl WindowChrome {
    fn observe_modifiers(&mut self, event: &PresentationEvent) {
        self.modifiers = match event {
            PresentationEvent::KeyDown {
                virtual_key,
                modifiers,
                ..
            } if *virtual_key == 0x11 => PresentationModifiers::new(
                true,
                modifiers.shift(),
                modifiers.alt(),
                modifiers.meta(),
            ),
            PresentationEvent::KeyUp {
                virtual_key,
                modifiers,
            } if *virtual_key == 0x11 => PresentationModifiers::new(
                false,
                modifiers.shift(),
                modifiers.alt(),
                modifiers.meta(),
            ),
            PresentationEvent::KeyDown { modifiers, .. }
            | PresentationEvent::KeyUp { modifiers, .. } => *modifiers,
            PresentationEvent::FocusLost => PresentationModifiers::NONE,
            _ => return,
        };
    }

    pub(super) const fn new(visible: bool) -> Self {
        Self {
            visible,
            open_menu: None,
            multi_series_dialog: None,
            calibration: None,
            show_series_preview: true,
            modifiers: PresentationModifiers::NONE,
            pointer_owner: PointerOwner::None,
            pressed_buttons: PressedButtons::EMPTY,
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
        if let Some(dialog) = &self.calibration {
            dialog.render(framebuffer)?;
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
        self.cancel_pointer_capture();
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
        self.pressed_buttons = PressedButtons::EMPTY;
    }

    pub(super) fn cancel_slice_drag(&mut self) {
        if matches!(self.pointer_owner, PointerOwner::Slice(_)) {
            self.pointer_owner = PointerOwner::Chrome;
        }
    }
}

#[cfg(test)]
#[path = "window_controls/tests.rs"]
mod tests;
