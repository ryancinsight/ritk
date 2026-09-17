//! Native DICOM series selection state and framebuffer overlay.
//!
//! The selector is a RITK presentation concern.  Métis only receives the
//! resulting framebuffer and the bounded keyboard events; DICOM discovery,
//! labels and UID selection remain in this crate.

use anyhow::{anyhow, Result};
use metis_platform::{Color, DisplayScale, Framebuffer, Rect};
use metis_ui_lang::{DisplayCommand, DisplayList};
use std::path::{Path, PathBuf};

use crate::dicom::series_tree::{SeriesEntryView, SeriesTree};

const KEY_ESCAPE: u32 = 0x1b;
const KEY_RETURN: u32 = 0x0d;
const KEY_HOME: u32 = 0x24;
const KEY_END: u32 = 0x23;
const KEY_UP: u32 = 0x26;
const KEY_DOWN: u32 = 0x28;
const KEY_ONE: u32 = 0x31;
const KEY_NINE: u32 = 0x39;
const MAX_VISIBLE_OPTIONS: usize = 12;
const ROW_HEIGHT: i32 = 24;
const MARGIN: i32 = 24;
const OVERLAY_WIDTH: i32 = 900;
const OVERLAY_BACKGROUND: Color = Color::rgba(9, 14, 24, 244);
const OVERLAY_BORDER: Color = Color::rgba(96, 144, 192, 255);
const OVERLAY_TEXT: Color = Color::rgba(240, 244, 248, 255);
const OVERLAY_MUTED: Color = Color::rgba(168, 188, 208, 255);
const OVERLAY_SELECTED: Color = Color::rgba(142, 220, 170, 255);
const OVERLAY_WARNING: Color = Color::rgba(255, 190, 120, 255);

/// One series choice retained by the native selector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SeriesChoice {
    /// Series Instance UID used for the exact RITK load.
    pub(crate) uid: Box<str>,
    /// RITK-owned human-readable label.
    pub(crate) label: Box<str>,
}

/// Interactive selector shown when a folder contains multiple series.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SeriesSelection {
    path: PathBuf,
    choices: Box<[SeriesChoice]>,
    selected: usize,
    notice: Option<Box<str>>,
}

/// Result of reducing one key event while the selector is visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectionAction {
    /// The selected row changed and the framebuffer should be repainted.
    Changed,
    /// The selector was canceled.
    Canceled,
    /// The selected row was confirmed.
    Confirmed,
    /// The key has no selector meaning.
    Ignored,
}

impl SeriesSelection {
    /// Build a selector from a discovered RITK series tree.
    pub(crate) fn from_tree(path: &Path, tree: &SeriesTree<'static>) -> Result<Self> {
        let total = tree.total_series();
        if total < 2 {
            return Err(anyhow!(
                "native series selector requires at least two discovered series"
            ));
        }
        let mut choices = Vec::new();
        choices
            .try_reserve_exact(total)
            .map_err(|_| anyhow!("native series selector allocation failed"))?;
        for series in tree.iter_series() {
            choices.push(SeriesChoice {
                uid: series.series_uid().to_owned().into_boxed_str(),
                label: series.display_label().into_boxed_str(),
            });
        }
        Ok(Self {
            path: path.to_path_buf(),
            choices: choices.into_boxed_slice(),
            selected: 0,
            notice: None,
        })
    }

    /// Return the number of discovered choices.
    pub(crate) const fn len(&self) -> usize {
        self.choices.len()
    }

    /// Return the currently highlighted row.
    #[cfg(test)]
    pub(crate) const fn selected(&self) -> usize {
        self.selected
    }

    /// Reduce one RITK presentation key event.
    pub(crate) fn handle_key(&mut self, virtual_key: u32, repeated: bool) -> SelectionAction {
        if repeated {
            return SelectionAction::Ignored;
        }
        match virtual_key {
            KEY_ESCAPE => SelectionAction::Canceled,
            KEY_RETURN => SelectionAction::Confirmed,
            KEY_UP if self.selected > 0 => {
                self.selected = self.selected.saturating_sub(1);
                SelectionAction::Changed
            }
            KEY_DOWN if self.selected + 1 < self.choices.len() => {
                self.selected = self.selected.saturating_add(1);
                SelectionAction::Changed
            }
            KEY_HOME => {
                if self.selected != 0 {
                    self.selected = 0;
                    SelectionAction::Changed
                } else {
                    SelectionAction::Ignored
                }
            }
            KEY_END => {
                let last = self.choices.len().saturating_sub(1);
                if self.selected != last {
                    self.selected = last;
                    SelectionAction::Changed
                } else {
                    SelectionAction::Ignored
                }
            }
            key if (KEY_ONE..=KEY_NINE).contains(&key) => {
                let Ok(index) = usize::try_from(key - KEY_ONE) else {
                    return SelectionAction::Ignored;
                };
                if index < self.choices.len() && index != self.selected {
                    self.selected = index;
                    SelectionAction::Changed
                } else {
                    SelectionAction::Ignored
                }
            }
            _ => SelectionAction::Ignored,
        }
    }

    pub(crate) fn selected_request(&self) -> (PathBuf, Box<str>) {
        let choice = self
            .choices
            .get(self.selected)
            .expect("invariant: selector always has a selected choice");
        (self.path.clone(), choice.uid.clone())
    }

    pub(crate) fn set_notice(&mut self, message: impl Into<Box<str>>) {
        self.notice = Some(message.into());
    }

    /// Append the selector to a framebuffer as a bounded display list.
    pub(crate) fn render_to(&self, framebuffer: &mut Framebuffer) -> Result<()> {
        let width = i32::try_from(framebuffer.width())
            .map_err(|_| anyhow!("native selector surface width exceeds i32"))?;
        let height = i32::try_from(framebuffer.height())
            .map_err(|_| anyhow!("native selector surface height exceeds i32"))?;
        let box_width = OVERLAY_WIDTH.min(width.saturating_sub(MARGIN * 2));
        let visible = self.choices.len().min(MAX_VISIBLE_OPTIONS);
        let box_height = 108_i32
            .checked_add(
                ROW_HEIGHT
                    .checked_mul(
                        i32::try_from(visible)
                            .map_err(|_| anyhow!("native selector row count exceeds i32"))?,
                    )
                    .ok_or_else(|| anyhow!("native selector height overflows i32"))?,
            )
            .and_then(|value| value.checked_add(40))
            .ok_or_else(|| anyhow!("native selector height overflows i32"))?;
        let x = (width.saturating_sub(box_width)) / 2;
        let y = (height.saturating_sub(box_height)) / 2;
        let mut display = DisplayList::default();
        push(
            &mut display,
            DisplayCommand::FillRect {
                rect: Rect::new(x, y, box_width, box_height),
                color: OVERLAY_BACKGROUND,
            },
        )?;
        push(
            &mut display,
            DisplayCommand::DrawBorder {
                rect: Rect::new(x, y, box_width, box_height),
                width: 2,
                color: OVERLAY_BORDER,
            },
        )?;
        push(
            &mut display,
            DisplayCommand::DrawText {
                text: format!("Select DICOM series ({} found)", self.choices.len()),
                x: x + MARGIN,
                y: y + 14,
                color: OVERLAY_TEXT,
                scale: 2,
                display_scale: DisplayScale::ONE,
            },
        )?;
        if let Some(notice) = &self.notice {
            push(
                &mut display,
                DisplayCommand::DrawText {
                    text: notice.to_string(),
                    x: x + MARGIN,
                    y: y + 40,
                    color: OVERLAY_WARNING,
                    scale: 1,
                    display_scale: DisplayScale::ONE,
                },
            )?;
        }
        let start = self
            .selected
            .saturating_sub(visible / 2)
            .min(self.choices.len().saturating_sub(visible));
        for (row, choice) in self.choices[start..start + visible].iter().enumerate() {
            let index = start + row;
            let row_y = y
                .checked_add(72)
                .and_then(|value| {
                    value.checked_add(ROW_HEIGHT.checked_mul(i32::try_from(row).ok()?)?)
                })
                .ok_or_else(|| anyhow!("native selector row position overflows i32"))?;
            let selected = index == self.selected;
            let marker = if selected { ">" } else { " " };
            push(
                &mut display,
                DisplayCommand::DrawText {
                    text: format!("{marker} {}. {}", index + 1, choice.label),
                    x: x + MARGIN,
                    y: row_y,
                    color: if selected {
                        OVERLAY_SELECTED
                    } else {
                        OVERLAY_TEXT
                    },
                    scale: 1,
                    display_scale: DisplayScale::ONE,
                },
            )?;
        }
        let footer_y = y
            .checked_add(box_height)
            .and_then(|value| value.checked_sub(28))
            .ok_or_else(|| anyhow!("native selector footer position underflows i32"))?;
        push(
            &mut display,
            DisplayCommand::DrawText {
                text: "Arrow keys or 1-9 select · Enter opens · Escape cancels".to_owned(),
                x: x + MARGIN,
                y: footer_y,
                color: OVERLAY_MUTED,
                scale: 1,
                display_scale: DisplayScale::ONE,
            },
        )?;
        display.render_to(framebuffer);
        Ok(())
    }
}

fn push(display: &mut DisplayList, command: DisplayCommand) -> Result<()> {
    display
        .commands
        .try_reserve(1)
        .map_err(|_| anyhow!("native selector display allocation failed"))?;
    display.commands.push(command);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dicom::loader::tests::fixtures;
    use tempfile::tempdir;

    #[test]
    fn selector_orders_and_reduces_bounded_keys() {
        let root = tempdir().expect("selector fixture root");
        fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID)
            .expect("write primary series");
        fixtures::write_study(root.path(), "MR", "2.25.20260905002")
            .expect("write secondary series");
        let tree = crate::dicom::loader::scan_folder_for_series(root.path())
            .expect("scan selector fixture");
        let mut selector = SeriesSelection::from_tree(root.path(), &tree).expect("selector");
        assert_eq!(selector.len(), 2);
        assert_eq!(selector.selected(), 0);
        assert_eq!(selector.handle_key(0x28, false), SelectionAction::Changed);
        assert_eq!(selector.selected(), 1);
        assert_eq!(selector.handle_key(0x24, false), SelectionAction::Changed);
        assert_eq!(selector.selected(), 0);
        assert_eq!(selector.handle_key(0x23, false), SelectionAction::Changed);
        assert_eq!(selector.selected(), 1);
        assert_eq!(selector.handle_key(0x28, true), SelectionAction::Ignored);
        assert_eq!(selector.handle_key(0x0d, false), SelectionAction::Confirmed);
        let (_, uid) = selector.selected_request();
        assert_eq!(uid.as_ref(), "2.25.20260905002");
    }

    #[test]
    fn selector_overlay_contains_selection_and_controls() {
        let root = tempdir().expect("selector fixture root");
        fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID)
            .expect("write primary series");
        fixtures::write_study(root.path(), "MR", "2.25.20260905002")
            .expect("write secondary series");
        let tree = crate::dicom::loader::scan_folder_for_series(root.path())
            .expect("scan selector fixture");
        let selector = SeriesSelection::from_tree(root.path(), &tree).expect("selector");
        let mut framebuffer = Framebuffer::new(1_280, 800).expect("selector framebuffer");
        selector
            .render_to(&mut framebuffer)
            .expect("selector overlay");
        assert_ne!(
            framebuffer.get_pixel(640, 400),
            Color::BLACK,
            "selector overlay changes the center of the framebuffer"
        );
    }
}
