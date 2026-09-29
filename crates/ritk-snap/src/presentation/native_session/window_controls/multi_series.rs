//! Filter and select series from the open study before assigning panels.

mod render;

use super::super::series_browser::SeriesBrowser;
use super::series::{patient_label, rect_contains, study_label};
use crate::presentation::{PointerButton, PresentationEvent};
use anyhow::{anyhow, Result};
use arrayvec::{ArrayString, ArrayVec};
use metis_platform::{Color, Rect};

const MAX_FILTER_BYTES: usize = 128;
const MAX_SELECTION: usize = super::super::layout::MAX_GRID_PANELS;
const DIALOG_WIDTH: i32 = 760;
const DIALOG_HEIGHT: i32 = 560;
const MIN_WINDOW_WIDTH: i32 = 380;
const MIN_WINDOW_HEIGHT: i32 = 340;
const ROW_HEIGHT: i32 = 32;
const LIST_TOP: i32 = 84;
const LIST_HEADER_HEIGHT: i32 = 24;
const FOOTER_HEIGHT: i32 = 64;
const PANEL: Color = Color::rgb(33, 39, 47);
const BORDER: Color = Color::rgb(91, 106, 120);
const ROW: Color = Color::rgb(42, 49, 58);
const FOCUSED_ROW: Color = Color::rgb(44, 96, 124);
const SELECTED: Color = Color::rgb(66, 148, 185);
const TEXT: Color = Color::rgb(231, 237, 243);
const MUTED: Color = Color::rgb(166, 181, 195);
const WARNING: Color = Color::rgb(255, 194, 118);
const OVERLAY: Color = Color::rgba(4, 7, 11, 176);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DialogAction {
    Confirm,
    Cancel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DialogEvent {
    pub(super) repaint: bool,
    pub(super) action: Option<DialogAction>,
}

#[derive(Clone)]
pub(super) struct MultiSeriesDialog {
    filter: ArrayString<MAX_FILTER_BYTES>,
    matches: Vec<usize>,
    selected: ArrayVec<usize, MAX_SELECTION>,
    cursor: usize,
    first_visible: usize,
    selection_limit_reached: bool,
}

impl MultiSeriesDialog {
    pub(super) fn new(browser: &SeriesBrowser) -> Result<Self> {
        let mut dialog = Self {
            filter: ArrayString::new(),
            matches: Vec::new(),
            selected: ArrayVec::new(),
            cursor: 0,
            first_visible: 0,
            selection_limit_reached: false,
        };
        dialog.rebuild_matches(browser)?;
        Ok(dialog)
    }

    pub(super) fn selected(&self) -> &ArrayVec<usize, MAX_SELECTION> {
        &self.selected
    }

    pub(super) fn handle_event(
        &mut self,
        event: &PresentationEvent,
        width: u32,
        height: u32,
        browser: &SeriesBrowser,
        control_down: bool,
    ) -> Result<DialogEvent> {
        let Some(geometry) = DialogGeometry::new(width, height)? else {
            return Ok(DialogEvent {
                repaint: false,
                action: None,
            });
        };
        match event {
            PresentationEvent::KeyDown {
                virtual_key,
                repeated,
                ..
            } if !repeated => match *virtual_key {
                0x1b => Ok(action(DialogAction::Cancel)),
                0x0d => {
                    if let Some(index) = self.matches.first().copied() {
                        self.set_single(index);
                        Ok(action(DialogAction::Confirm))
                    } else {
                        Ok(changed(false))
                    }
                }
                0x08 => {
                    if self.filter.pop().is_some() {
                        self.rebuild_matches(browser)?;
                        Ok(changed(true))
                    } else {
                        Ok(changed(false))
                    }
                }
                0x26 => Ok(changed(self.move_cursor(-1, geometry))),
                0x28 => Ok(changed(self.move_cursor(1, geometry))),
                0x20 => {
                    if let Some(index) = self.matches.get(self.cursor).copied() {
                        self.toggle(index);
                        Ok(changed(true))
                    } else {
                        Ok(changed(false))
                    }
                }
                _ => Ok(changed(false)),
            },
            PresentationEvent::TextInput { character } if !character.is_control() => {
                if self.filter.len().saturating_add(character.len_utf8()) > MAX_FILTER_BYTES {
                    return Ok(changed(false));
                }
                self.filter
                    .try_push(*character)
                    .map_err(|_| anyhow!("bounded series filter could not accept a character"))?;
                self.rebuild_matches(browser)?;
                Ok(changed(true))
            }
            PresentationEvent::PointerDown {
                x,
                y,
                button: PointerButton::Left,
            } => {
                if let Some(index) = row_at(geometry, &self.matches, self.first_visible, *x, *y) {
                    self.cursor = index;
                    let series_index = self.matches[index];
                    if control_down {
                        self.toggle(series_index);
                    } else {
                        self.set_single(series_index);
                    }
                    Ok(changed(true))
                } else if rect_contains(geometry.open, *x, *y) {
                    if self.selected.is_empty() && !self.matches.is_empty() {
                        self.toggle(self.matches[0]);
                    }
                    Ok(if self.selected.is_empty() {
                        changed(false)
                    } else {
                        action(DialogAction::Confirm)
                    })
                } else if rect_contains(geometry.cancel, *x, *y)
                    || !rect_contains(geometry.dialog, *x, *y)
                {
                    Ok(action(DialogAction::Cancel))
                } else {
                    Ok(changed(false))
                }
            }
            PresentationEvent::PointerWheel { x, y, delta_y, .. }
                if rect_contains(geometry.list, *x, *y) =>
            {
                let delta = if *delta_y > 0.0 {
                    -3
                } else if *delta_y < 0.0 {
                    3
                } else {
                    0
                };
                Ok(changed(self.scroll(delta, geometry)))
            }
            _ => Ok(changed(false)),
        }
    }

    fn visible_rows(&self, geometry: DialogGeometry) -> usize {
        usize::try_from(geometry.list.height / ROW_HEIGHT)
            .expect("invariant: positive dialog list height fits in usize")
    }

    fn toggle(&mut self, series_index: usize) {
        if let Some(position) = self
            .selected
            .iter()
            .position(|index| *index == series_index)
        {
            self.selected.remove(position);
            self.selection_limit_reached = false;
        } else {
            self.selection_limit_reached = self.selected.try_push(series_index).is_err();
        }
    }

    fn set_single(&mut self, series_index: usize) {
        self.selected.clear();
        self.selected
            .try_push(series_index)
            .expect("invariant: one selection fits the bounded panel capacity");
        self.selection_limit_reached = false;
    }

    fn rebuild_matches(&mut self, browser: &SeriesBrowser) -> Result<()> {
        self.matches.clear();
        self.matches
            .try_reserve(browser.len())
            .map_err(|_| anyhow!("series filter allocation failed"))?;
        let query = self.filter.to_lowercase();
        for index in 0..browser.len() {
            let choice = browser
                .choice(index)
                .ok_or_else(|| anyhow!("series catalog changed while filtering"))?;
            let patient = patient_label(choice)?;
            let study = study_label(choice)?;
            let searchable = format!(
                "{} {} {} {} {} {} {}",
                choice.description,
                choice.modality,
                patient,
                study,
                choice.patient_number,
                choice.study_number,
                choice.instance_count
            );
            if searchable.to_lowercase().contains(&query) {
                self.matches.push(index);
            }
        }
        self.cursor = 0;
        self.first_visible = 0;
        self.selection_limit_reached = false;
        Ok(())
    }

    fn move_cursor(&mut self, delta: i32, geometry: DialogGeometry) -> bool {
        if self.matches.is_empty() {
            return false;
        }
        let previous = self.cursor;
        self.cursor = if delta < 0 {
            self.cursor.saturating_sub(1)
        } else {
            self.cursor
                .saturating_add(1)
                .min(self.matches.len().saturating_sub(1))
        };
        self.keep_cursor_visible(geometry);
        self.cursor != previous
    }

    fn keep_cursor_visible(&mut self, geometry: DialogGeometry) {
        let visible = self.visible_rows(geometry).max(1);
        if self.cursor < self.first_visible {
            self.first_visible = self.cursor;
        } else if self.cursor >= self.first_visible.saturating_add(visible) {
            self.first_visible = self.cursor.saturating_add(1).saturating_sub(visible);
        }
    }

    fn scroll(&mut self, delta: i32, geometry: DialogGeometry) -> bool {
        let visible = self.visible_rows(geometry);
        let maximum = self.matches.len().saturating_sub(visible);
        let previous = self.first_visible;
        self.first_visible = if delta < 0 {
            previous.saturating_sub(
                usize::try_from(delta.unsigned_abs())
                    .expect("invariant: bounded dialog scroll fits in usize"),
            )
        } else {
            previous
                .saturating_add(
                    usize::try_from(delta)
                        .expect("invariant: nonnegative dialog scroll fits in usize"),
                )
                .min(maximum)
        };
        self.first_visible != previous
    }
}

#[derive(Clone, Copy)]
struct DialogGeometry {
    dialog: Rect,
    filter: Rect,
    list: Rect,
    cancel: Rect,
    open: Rect,
}

impl DialogGeometry {
    fn new(width: u32, height: u32) -> Result<Option<Self>> {
        let width = i32::try_from(width).map_err(|_| anyhow!("dialog width exceeds i32"))?;
        let height = i32::try_from(height).map_err(|_| anyhow!("dialog height exceeds i32"))?;
        if width < MIN_WINDOW_WIDTH || height < MIN_WINDOW_HEIGHT {
            return Ok(None);
        }
        let dialog_width = DIALOG_WIDTH.min(width.saturating_sub(32));
        let dialog_height = DIALOG_HEIGHT.min(height.saturating_sub(32));
        let dialog = Rect::new(
            width.saturating_sub(dialog_width) / 2,
            height.saturating_sub(dialog_height) / 2,
            dialog_width,
            dialog_height,
        );
        let list_height = dialog.height - LIST_TOP - LIST_HEADER_HEIGHT - FOOTER_HEIGHT - 20;
        if list_height < ROW_HEIGHT {
            return Ok(None);
        }
        let list = Rect::new(
            dialog.x + 20,
            dialog.y + LIST_TOP + LIST_HEADER_HEIGHT,
            dialog.width - 40,
            list_height,
        );
        let button_y = dialog.y + dialog.height - FOOTER_HEIGHT + 14;
        let filter = Rect::new(
            dialog.x + 20,
            button_y,
            dialog.width.saturating_sub(260).max(80),
            32,
        );
        let cancel = Rect::new(dialog.x + dialog.width - 190, button_y, 76, 32);
        let open = Rect::new(dialog.x + dialog.width - 102, button_y, 82, 32);
        Ok(Some(Self {
            dialog,
            filter,
            list,
            cancel,
            open,
        }))
    }
}

fn row_at(
    geometry: DialogGeometry,
    matches: &[usize],
    first_visible: usize,
    x: f64,
    y: f64,
) -> Option<usize> {
    if !rect_contains(geometry.list, x, y) {
        return None;
    }
    let visible = usize::try_from(geometry.list.height / ROW_HEIGHT).ok()?;
    for row in 0..visible.min(matches.len().saturating_sub(first_visible)) {
        let row_i32 = i32::try_from(row).ok()?;
        let row_y = geometry
            .list
            .y
            .checked_add(row_i32.checked_mul(ROW_HEIGHT)?)?;
        if rect_contains(
            Rect::new(geometry.list.x, row_y, geometry.list.width, ROW_HEIGHT),
            x,
            y,
        ) {
            return first_visible.checked_add(row);
        }
    }
    None
}

fn changed(repaint: bool) -> DialogEvent {
    DialogEvent {
        repaint,
        action: None,
    }
}

fn action(action: DialogAction) -> DialogEvent {
    DialogEvent {
        repaint: true,
        action: Some(action),
    }
}

#[cfg(test)]
#[path = "multi_series/tests.rs"]
mod tests;
