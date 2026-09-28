//! RITK-owned controls rendered in the visible Métis native client area.

mod layout;
mod series;
use self::layout::{ChromeGeometry, ChromeLayout};
use super::layout::{NativeViewport, ViewportArea, WorkspaceLayout};
use super::series_browser::SeriesBrowser;
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationEvent, PresentationFrame};
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
    Exit,
    SelectSeries(usize),
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

pub(super) struct WindowChrome {
    visible: bool,
    open_menu: Option<Menu>,
    show_series_preview: bool,
    pointer_owner: PointerOwner,
    pressed_buttons: u8,
}

impl WindowChrome {
    pub(super) const fn new(visible: bool) -> Self {
        Self {
            visible,
            open_menu: None,
            show_series_preview: true,
            pointer_owner: PointerOwner::None,
            pressed_buttons: 0,
        }
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
        browser: Option<&SeriesBrowser>,
        workspace_layout: WorkspaceLayout,
        active_panel: usize,
        displayed_series: &[Option<usize>],
    ) -> Result<()> {
        if !self.visible {
            return Ok(());
        }
        ChromeLayout::new(
            framebuffer.width(),
            framebuffer.height(),
            self.open_menu,
            app,
            self.show_series_preview,
            workspace_layout,
        )?
        .render(
            framebuffer,
            app,
            series_previews,
            browser,
            workspace_layout,
            active_panel,
            displayed_series,
        )
    }

    pub(super) fn handle_event(
        &mut self,
        event: &PresentationEvent,
        width: u32,
        height: u32,
        app: &SnapApp,
        browser: &mut Option<SeriesBrowser>,
        workspace_layout: WorkspaceLayout,
        viewports: &[NativeViewport],
    ) -> Result<WindowChromeEvent> {
        if !self.visible {
            return Ok(WindowChromeEvent::passed());
        }
        if matches!(
            event,
            PresentationEvent::KeyDown {
                virtual_key: 0x1b,
                repeated: false,
                ..
            }
        ) && self.open_menu.take().is_some()
        {
            return Ok(WindowChromeEvent::consumed(true));
        }

        let layout = ChromeLayout::new(
            width,
            height,
            self.open_menu,
            app,
            self.show_series_preview,
            workspace_layout,
        )?;
        match event {
            PresentationEvent::PointerDown { x, y, button } => {
                if self.pointer_owner == PointerOwner::None && *button == PointerButton::Left {
                    if let Some(index) = layout.series_index_at(browser.as_ref(), *x, *y) {
                        self.pointer_owner = PointerOwner::Series(index);
                        self.press(*button);
                        return Ok(WindowChromeEvent::consumed(false));
                    }
                }
                if self.pointer_owner == PointerOwner::Chrome {
                    self.press(*button);
                    return Ok(WindowChromeEvent::consumed(false));
                }
                if matches!(self.pointer_owner, PointerOwner::Series(_)) {
                    self.press(*button);
                    return Ok(WindowChromeEvent::consumed(false));
                }
                if self.pointer_owner == PointerOwner::Pane {
                    self.press(*button);
                    return Ok(WindowChromeEvent::passed());
                }
                if layout.owns_pointer(*x, *y) || self.open_menu.is_some() {
                    self.pointer_owner = PointerOwner::Chrome;
                    self.press(*button);
                    if *button != PointerButton::Left {
                        return Ok(WindowChromeEvent::consumed(false));
                    }
                    let action = layout.action_at(*x, *y, browser.as_ref());
                    if let Some(WindowAction::OpenMenu(menu)) = action {
                        self.open_menu = (self.open_menu != Some(menu)).then_some(menu);
                        return Ok(WindowChromeEvent::consumed(true));
                    }
                    if let Some(action) = action {
                        self.open_menu = None;
                        return Ok(WindowChromeEvent {
                            consumed: true,
                            repaint: true,
                            action: Some(action),
                        });
                    }
                    let repaint = self.open_menu.take().is_some();
                    return Ok(WindowChromeEvent::consumed(repaint));
                }
                self.pointer_owner = PointerOwner::Pane;
                self.press(*button);
                Ok(WindowChromeEvent::passed())
            }
            PresentationEvent::PointerMove { x, y } => Ok(
                if matches!(self.pointer_owner, PointerOwner::Series(_))
                    || self.pointer_owner == PointerOwner::Chrome
                    || self.open_menu.is_some()
                    || self.pointer_owner == PointerOwner::None && layout.owns_pointer(*x, *y)
                {
                    WindowChromeEvent::consumed(false)
                } else {
                    WindowChromeEvent::passed()
                },
            ),
            PresentationEvent::PointerUp { x, y, button } => {
                let (consumed, action) = match self.pointer_owner {
                    PointerOwner::Chrome => (true, None),
                    PointerOwner::Pane => (false, None),
                    PointerOwner::None => (layout.owns_pointer(*x, *y), None),
                    PointerOwner::Series(index) => {
                        let action = if *button != PointerButton::Left {
                            None
                        } else {
                            let target_panel = viewports
                                .iter()
                                .position(|viewport| viewport.contains(*x, *y));
                            if let Some(panel_index) = target_panel {
                                Some(if workspace_layout.is_grid() {
                                    WindowAction::AssignSeries {
                                        series_index: index,
                                        panel_index,
                                    }
                                } else {
                                    WindowAction::SelectSeries(index)
                                })
                            } else if layout.series_index_at(browser.as_ref(), *x, *y)
                                == Some(index)
                            {
                                Some(WindowAction::SelectSeries(index))
                            } else {
                                None
                            }
                        };
                        (true, action)
                    }
                };
                self.release(*button);
                Ok(WindowChromeEvent {
                    consumed,
                    repaint: action.is_some(),
                    action,
                })
            }
            PresentationEvent::PointerCancel { x, y, .. } => {
                let consumed = match self.pointer_owner {
                    PointerOwner::Chrome => true,
                    PointerOwner::Pane => false,
                    PointerOwner::None => layout.owns_pointer(*x, *y),
                    PointerOwner::Series(_) => true,
                };
                self.pointer_owner = PointerOwner::None;
                self.pressed_buttons = 0;
                Ok(if consumed {
                    WindowChromeEvent::consumed(false)
                } else {
                    WindowChromeEvent::passed()
                })
            }
            PresentationEvent::PointerWheel {
                x,
                y,
                delta_x,
                delta_y,
                ..
            } => {
                if layout.series_contains(*x, *y) {
                    let direction = if *delta_x != 0.0 { *delta_x } else { *delta_y };
                    let delta = if direction > 0.0 {
                        -3
                    } else if direction < 0.0 {
                        3
                    } else {
                        0
                    };
                    let changed = browser.as_mut().is_some_and(|series_browser| {
                        series_browser.scroll_series(delta, layout.visible_series())
                    });
                    Ok(WindowChromeEvent::consumed(changed))
                } else if self.open_menu.is_some() || layout.owns_pointer(*x, *y) {
                    Ok(WindowChromeEvent::consumed(false))
                } else {
                    Ok(WindowChromeEvent::passed())
                }
            }
            PresentationEvent::FocusLost => {
                let repaint = self.open_menu.take().is_some();
                self.pointer_owner = PointerOwner::None;
                self.pressed_buttons = 0;
                Ok(WindowChromeEvent {
                    consumed: false,
                    repaint,
                    action: None,
                })
            }
            _ => Ok(WindowChromeEvent::passed()),
        }
    }

    fn press(&mut self, button: PointerButton) {
        self.pressed_buttons |= pointer_button_bit(button);
    }

    fn release(&mut self, button: PointerButton) {
        self.pressed_buttons &= !pointer_button_bit(button);
        if self.pressed_buttons == 0 {
            self.pointer_owner = PointerOwner::None;
        }
    }

    pub(super) fn toggle_series_preview(&mut self) {
        self.show_series_preview = !self.show_series_preview;
        self.open_menu = None;
    }
}

fn pointer_button_bit(button: PointerButton) -> u8 {
    match button {
        PointerButton::Left => 1,
        PointerButton::Right => 2,
        PointerButton::Middle => 4,
        PointerButton::X1 => 8,
        PointerButton::X2 => 16,
    }
}

#[cfg(test)]
#[path = "window_controls/tests.rs"]
mod tests;
