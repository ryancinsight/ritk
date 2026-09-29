use super::*;
use crate::dicom::loader::{scan_folder_for_series, tests::fixtures};
use metis_platform::Framebuffer;

fn dialog_with_series(count: usize) -> (MultiSeriesDialog, SeriesBrowser, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("series root");
    for index in 0..count {
        let modality = if index % 2 == 0 { "MR" } else { "CT" };
        fixtures::write_study(root.path(), modality, &format!("2.25.20260906{index:04}"))
            .expect("write DICOM series");
    }
    let tree = scan_folder_for_series(root.path()).expect("scan study");
    let browser = SeriesBrowser::from_tree(&tree, None).expect("build series catalog");
    let dialog = MultiSeriesDialog::new(&browser).expect("build series picker");
    (dialog, browser, root)
}

#[test]
fn enter_opens_the_first_visible_series_after_filtering_the_list() {
    let (mut dialog, browser, _root) = dialog_with_series(3);
    let previous_selection = (0..browser.len())
        .find(|index| {
            browser
                .choice(*index)
                .is_some_and(|choice| choice.modality.as_ref() != "CT")
        })
        .expect("study contains a series outside the CT filter");
    dialog.set_single(previous_selection);
    dialog.filter.try_push('C').expect("append filter");
    dialog.filter.try_push('T').expect("append filter");
    dialog.rebuild_matches(&browser).expect("filter series");
    assert_eq!(dialog.matches.len(), 1);
    assert_ne!(dialog.matches[0], previous_selection);

    let event = dialog
        .handle_event(
            &PresentationEvent::KeyDown {
                virtual_key: 0x0d,
                repeated: false,
                modifiers: crate::presentation::PresentationModifiers::NONE,
            },
            1_280,
            800,
            &browser,
            false,
        )
        .expect("confirm selections");
    assert_eq!(event.action, Some(DialogAction::Confirm));
    assert_eq!(dialog.selected.as_slice(), &[dialog.matches[0]]);
    assert_ne!(dialog.selected[0], previous_selection);
}

#[test]
fn enter_does_not_confirm_a_hidden_selection_when_the_filter_has_no_matches() {
    let (mut dialog, browser, _root) = dialog_with_series(2);
    dialog.set_single(0);
    dialog
        .filter
        .try_push_str("no matching series")
        .expect("fit filter text");
    dialog.rebuild_matches(&browser).expect("filter series");
    assert!(dialog.matches.is_empty());

    let event = dialog
        .handle_event(
            &PresentationEvent::KeyDown {
                virtual_key: 0x0d,
                repeated: false,
                modifiers: crate::presentation::PresentationModifiers::NONE,
            },
            1_280,
            800,
            &browser,
            false,
        )
        .expect("do not open a hidden series");

    assert_eq!(event.action, None);
    assert!(dialog.selected.is_empty());
}

#[test]
fn plain_click_selects_one_and_ctrl_click_extends_the_selection() {
    let (mut dialog, browser, _root) = dialog_with_series(3);
    let first = dialog
        .handle_event(
            &PresentationEvent::PointerDown {
                x: 300.0,
                y: 232.0,
                button: PointerButton::Left,
            },
            1_280,
            800,
            &browser,
            false,
        )
        .expect("select first row");
    assert_eq!(first.action, None);
    assert_eq!(dialog.selected.as_slice(), &[0]);

    dialog
        .handle_event(
            &PresentationEvent::PointerDown {
                x: 300.0,
                y: 264.0,
                button: PointerButton::Left,
            },
            1_280,
            800,
            &browser,
            true,
        )
        .expect("extend selection with Control");
    assert_eq!(dialog.selected.as_slice(), &[0, 1]);

    dialog
        .handle_event(
            &PresentationEvent::PointerDown {
                x: 300.0,
                y: 264.0,
                button: PointerButton::Left,
            },
            1_280,
            800,
            &browser,
            false,
        )
        .expect("replace selection without Control");
    assert_eq!(dialog.selected.as_slice(), &[1]);
}

#[test]
fn picker_exposes_filter_result_and_capacity_without_truncation() {
    let (mut dialog, _browser, _root) = dialog_with_series(MAX_SELECTION + 1);
    for index in 0..MAX_SELECTION + 1 {
        dialog.toggle(index);
    }
    assert_eq!(dialog.selected.len(), MAX_SELECTION);
    assert!(dialog.selection_limit_reached);
    assert_eq!(dialog.matches.len(), MAX_SELECTION + 1);
}

#[test]
fn enter_opens_the_first_filtered_series_when_nothing_is_selected() {
    let (mut dialog, browser, _root) = dialog_with_series(2);
    let event = dialog
        .handle_event(
            &PresentationEvent::KeyDown {
                virtual_key: 0x0d,
                repeated: false,
                modifiers: crate::presentation::PresentationModifiers::NONE,
            },
            1_280,
            800,
            &browser,
            false,
        )
        .expect("open the first matching series");
    assert_eq!(event.action, Some(DialogAction::Confirm));
    assert_eq!(dialog.selected.as_slice(), &[0]);
}

#[test]
fn enter_opens_every_series_selected_with_the_keyboard() {
    let (mut dialog, browser, _root) = dialog_with_series(3);
    fn press(dialog: &mut MultiSeriesDialog, browser: &SeriesBrowser, virtual_key: u32) {
        dialog
            .handle_event(
                &PresentationEvent::KeyDown {
                    virtual_key,
                    repeated: false,
                    modifiers: crate::presentation::PresentationModifiers::NONE,
                },
                1_280,
                800,
                browser,
                false,
            )
            .expect("navigate and select series");
    }

    press(&mut dialog, &browser, 0x20);
    assert_eq!(dialog.selected().as_slice(), &[0]);
    press(&mut dialog, &browser, 0x28);
    press(&mut dialog, &browser, 0x20);
    assert_eq!(dialog.selected().as_slice(), &[0, 1]);
    assert_eq!(dialog.cursor, 1);

    let event = dialog
        .handle_event(
            &PresentationEvent::KeyDown {
                virtual_key: 0x0d,
                repeated: false,
                modifiers: crate::presentation::PresentationModifiers::NONE,
            },
            1_280,
            800,
            &browser,
            false,
        )
        .expect("open the keyboard selection");

    assert_eq!(event.action, Some(DialogAction::Confirm));
    assert_eq!(dialog.selected().as_slice(), &[0, 1]);
}

#[test]
fn escape_cancels_without_committing_a_selection() {
    let (mut dialog, browser, _root) = dialog_with_series(2);
    let event = dialog
        .handle_event(
            &PresentationEvent::KeyDown {
                virtual_key: 0x1b,
                repeated: false,
                modifiers: crate::presentation::PresentationModifiers::NONE,
            },
            1_280,
            800,
            &browser,
            false,
        )
        .expect("cancel picker");
    assert_eq!(event.action, Some(DialogAction::Cancel));
    assert!(dialog.selected.is_empty());
}

#[test]
fn picker_renders_study_column_beneath_the_table_header() {
    let (dialog, browser, _root) = dialog_with_series(2);
    let mut framebuffer = Framebuffer::new(1_280, 800).expect("series picker framebuffer");
    let geometry = DialogGeometry::new(1_280, 800)
        .expect("series picker geometry")
        .expect("window can display picker");

    dialog
        .render(&mut framebuffer, &browser)
        .expect("draw grouped series picker");

    let header_y = geometry.list.y - LIST_HEADER_HEIGHT + 1;
    let header_x = geometry.list.x + 1;
    assert_eq!(
        framebuffer.get_pixel(header_x, header_y),
        render::TABLE_HEADER_BACKGROUND
    );
    assert_eq!(
        framebuffer.get_pixel(geometry.list.x + 3, geometry.list.y + 3),
        FOCUSED_ROW
    );
    assert_eq!(
        framebuffer.get_pixel(geometry.list.x + 3, geometry.list.y + 2 * ROW_HEIGHT + 2),
        render::LIST_BACKGROUND
    );
    assert_eq!(
        framebuffer.get_pixel(geometry.open.x + 1, geometry.open.y + 1),
        SELECTED
    );

    let study_left = geometry.list.x + 160;
    let study_right = study_left + 142;
    let row_top = geometry.list.y + 8;
    let row_bottom = geometry.list.y + ROW_HEIGHT - 2;
    let has_study_text = (row_top..row_bottom)
        .any(|y| (study_left..study_right).any(|x| framebuffer.get_pixel(x, y) != FOCUSED_ROW));
    assert!(has_study_text, "the first row must render its study label");
}
