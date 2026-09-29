//! Native-window pointer, keyboard and dialog input ownership.

use super::super::layout::{
    panel_header_action_at, NativeViewport, PanelHeaderAction, WorkspaceLayout,
};
use super::super::series_browser::SeriesBrowser;
use super::layout::ChromeLayout;
use super::multi_series::DialogAction;
use super::{PanelCloseKind, PointerOwner, WindowAction, WindowChrome, WindowChromeEvent};
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationEvent, PresentationModifiers};
use anyhow::{anyhow, Result};

impl WindowChrome {
    pub(in crate::presentation::native_session) fn handle_event(
        &mut self,
        event: &PresentationEvent,
        width: u32,
        height: u32,
        app: &SnapApp,
        browser: &mut Option<SeriesBrowser>,
        workspace_layout: WorkspaceLayout,
        maximized_panel: bool,
        viewports: &[NativeViewport],
    ) -> Result<WindowChromeEvent> {
        if !self.visible {
            return Ok(WindowChromeEvent::passed());
        }
        match event {
            PresentationEvent::KeyDown {
                virtual_key,
                modifiers,
                ..
            } => self.control_down = modifiers.ctrl() || *virtual_key == 0x11,
            PresentationEvent::KeyUp {
                virtual_key,
                modifiers,
            } => self.control_down = *virtual_key != 0x11 && modifiers.ctrl(),
            PresentationEvent::FocusLost => self.control_down = false,
            _ => {}
        }
        if let Some(dialog) = self.multi_series_dialog.as_mut() {
            let Some(series_browser) = browser.as_ref() else {
                self.multi_series_dialog = None;
                return Ok(WindowChromeEvent::consumed(true));
            };
            let result =
                dialog.handle_event(event, width, height, series_browser, self.control_down)?;
            let action = match result.action {
                Some(DialogAction::Confirm) => Some(WindowAction::LoadSelectedSeries),
                Some(DialogAction::Cancel) => {
                    self.multi_series_dialog = None;
                    None
                }
                None => None,
            };
            return Ok(WindowChromeEvent {
                consumed: true,
                repaint: result.repaint,
                action,
            });
        }
        if let PresentationEvent::KeyDown {
            virtual_key,
            repeated: false,
            modifiers,
        } = event
        {
            let action = match (*virtual_key, modifiers.ctrl(), modifiers.shift()) {
                (0x73, false, false) => Some(WindowAction::OpenSeriesPicker),
                (0x73, true, false) => Some(WindowAction::CloseActivePanel),
                (0x73, false, true) => Some(WindowAction::CloseAllPanels),
                (0x4d, true, false) => Some(WindowAction::ToggleActivePanel),
                (0x09, _, true) => Some(WindowAction::ActivatePreviousPanel),
                (0x09, _, false) => Some(WindowAction::ActivateNextPanel),
                _ => None,
            };
            if let Some(action) = action {
                return Ok(WindowChromeEvent {
                    consumed: true,
                    repaint: true,
                    action: Some(action),
                });
            }
        }
        if self.open_menu.is_none() {
            if let PresentationEvent::KeyDown {
                virtual_key,
                repeated: false,
                modifiers,
            } = event
            {
                if *modifiers == PresentationModifiers::NONE {
                    let index = match *virtual_key {
                        0x24 => browser
                            .as_ref()
                            .filter(|browser| browser.active_index() != 0)
                            .map(|_| 0),
                        0x23 => browser.as_ref().and_then(|browser| {
                            (browser.active_index().saturating_add(1) < browser.len())
                                .then_some(browser.len().saturating_sub(1))
                        }),
                        0x25 => browser
                            .as_ref()
                            .and_then(|browser| adjacent_series(browser, -1)),
                        0x27 => browser
                            .as_ref()
                            .and_then(|browser| adjacent_series(browser, 1)),
                        _ => None,
                    };
                    if let Some(index) = index {
                        return Ok(WindowChromeEvent {
                            consumed: true,
                            repaint: true,
                            action: Some(WindowAction::SelectSeries(index)),
                        });
                    }
                }
            }
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
                    let panel_action = (self.open_menu.is_none()
                        && (workspace_layout.is_grid() || maximized_panel))
                        .then(|| panel_header_action_at(viewports, *x, *y, maximized_panel))
                        .flatten();
                    if let Some((index, action)) = panel_action {
                        self.pointer_owner = PointerOwner::Chrome;
                        self.press(*button);
                        let action = match action {
                            PanelHeaderAction::Maximize => WindowAction::MaximizePanel(index),
                            PanelHeaderAction::Close => WindowAction::ClosePanel {
                                index,
                                kind: if self.control_down {
                                    PanelCloseKind::Clear
                                } else {
                                    PanelCloseKind::Close
                                },
                            },
                        };
                        return Ok(WindowChromeEvent {
                            consumed: true,
                            repaint: true,
                            action: Some(action),
                        });
                    }
                }
                if self.pointer_owner == PointerOwner::None && *button == PointerButton::Left {
                    if layout.popup_contains(*x, *y) {
                        self.pointer_owner = PointerOwner::Chrome;
                        self.press(*button);
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
                        return Ok(WindowChromeEvent::consumed(false));
                    }
                    let direction = match browser.as_ref() {
                        Some(series_browser) => {
                            layout.series_scroll_direction_at(series_browser, *x, *y)?
                        }
                        None => None,
                    };
                    if let Some(direction) = direction {
                        self.pointer_owner = PointerOwner::Chrome;
                        self.press(*button);
                        let page = i32::try_from(layout.visible_series())
                            .map_err(|_| anyhow!("visible series count exceeds i32"))?;
                        let page_size = usize::try_from(page)
                            .map_err(|_| anyhow!("visible series count is negative"))?;
                        let changed = browser.as_mut().is_some_and(|series_browser| {
                            series_browser.scroll_series(direction.saturating_mul(page), page_size)
                        });
                        return Ok(WindowChromeEvent::consumed(changed));
                    }
                    if let Some(index) = layout.series_index_at(browser.as_ref(), *x, *y) {
                        self.pointer_owner = PointerOwner::Series(index);
                        self.press(*button);
                        let repaint = self.open_menu.take().is_some();
                        return Ok(WindowChromeEvent::consumed(repaint));
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
                                Some(if self.control_down {
                                    WindowAction::OpenSeriesInNextPanel(index)
                                } else {
                                    WindowAction::SelectSeries(index)
                                })
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
                modifiers,
            } => {
                if layout.popup_contains(*x, *y) {
                    Ok(WindowChromeEvent::consumed(false))
                } else if layout.series_contains(*x, *y) {
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
                } else if delta_x.is_finite()
                    && *delta_x != 0.0
                    && *modifiers == PresentationModifiers::NONE
                    && viewports.iter().any(|viewport| viewport.contains(*x, *y))
                {
                    let direction = if *delta_x > 0.0 { 1 } else { -1 };
                    let action = browser
                        .as_ref()
                        .and_then(|browser| adjacent_series(browser, direction))
                        .map(WindowAction::SelectSeries);
                    Ok(match action {
                        Some(action) => WindowChromeEvent {
                            consumed: true,
                            repaint: true,
                            action: Some(action),
                        },
                        None => WindowChromeEvent::passed(),
                    })
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
}

fn adjacent_series(browser: &SeriesBrowser, direction: i32) -> Option<usize> {
    let active = browser.active_index();
    let next = if direction < 0 {
        active.checked_sub(1)?
    } else {
        active.checked_add(1)?
    };
    browser.choice(next).map(|_| next)
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
