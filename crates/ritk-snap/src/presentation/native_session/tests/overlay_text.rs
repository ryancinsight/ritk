//! Native overlay text: orientation letters, demographics, ruler, cursor readout.
//!
//! The native session paints its own clinical overlay through
//! [`super::super::layout::application_overlay`]; these tests pin the
//! overlay-text contract from RITK-SNAP-NATIVE-OVERLAY-TEXT-001 without a GPU.

use super::super::layout::{
    application_overlay, clinical_labels_for_view, scale_bar_for_scale, ClinicalOverlay,
};
use super::support::session;
use super::*;
use crate::ui::ViewTransform;
use metis_ui_lang::DisplayCommand;

const STANDARD_DIRECTION: [f64; 9] = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];

fn overlay_texts(commands: &[DisplayCommand]) -> Vec<&str> {
    commands
        .iter()
        .filter_map(|command| match command {
            DisplayCommand::DrawText { text, .. } => Some(text.as_ref()),
            _ => None,
        })
        .collect()
}

#[test]
fn native_overlay_text_shows_orientation_letters_for_known_direction() {
    let labels =
        clinical_labels_for_view(0, &STANDARD_DIRECTION, [512, 512], ViewTransform::default());
    assert_eq!(
        (labels.left, labels.right, labels.top, labels.bottom),
        ("R", "L", "A", "P")
    );
    let labels =
        clinical_labels_for_view(1, &STANDARD_DIRECTION, [512, 512], ViewTransform::default());
    assert_eq!(
        (labels.left, labels.right, labels.top, labels.bottom),
        ("R", "L", "I", "S")
    );
    let labels =
        clinical_labels_for_view(2, &STANDARD_DIRECTION, [512, 512], ViewTransform::default());
    assert_eq!(
        (labels.left, labels.right, labels.top, labels.bottom),
        ("A", "P", "I", "S")
    );
}

#[test]
fn native_overlay_text_ruler_matches_pixel_spacing() {
    assert_eq!(scale_bar_for_scale(2.0, 500.0), Some((50.0, 100.0)));
    assert_eq!(scale_bar_for_scale(1.0, 500.0), Some((100.0, 100.0)));
    assert_eq!(scale_bar_for_scale(0.2, 500.0), Some((500.0, 100.0)));
    assert_eq!(scale_bar_for_scale(0.0, 500.0), None);
    assert_eq!(scale_bar_for_scale(f64::NAN, 500.0), None);
    assert_eq!(scale_bar_for_scale(2.0, 10.0), None);
}

fn clinical() -> ClinicalOverlay<'static> {
    ClinicalOverlay {
        direction: STANDARD_DIRECTION,
        patient_name: Some("Test Patient"),
        patient_id: Some("ID-1"),
        series_description: Some("Test Series"),
        modality: Some("CT"),
        study_date: Some("20240101"),
        spacing: [0.5, 0.5, 1.0],
        zoom: 1.0,
        cursor_value: Some(142.0),
        anonymize: false,
    }
}

fn overlay_commands(clinical: &ClinicalOverlay) -> Vec<String> {
    let (session, _root) = session();
    let overlay = application_overlay(
        &session.views,
        &session.viewports,
        false,
        30.0,
        Some(clinical),
        &[],
        None,
    )
    .expect("compose the native overlay commands")
    .into_display_list();
    overlay_texts(overlay.commands())
        .into_iter()
        .map(str::to_owned)
        .collect()
}

#[test]
fn native_overlay_text_shows_demographics_corners() {
    let texts = overlay_commands(&clinical());
    for expected in [
        "Test Patient",
        "ID-1",
        "Test Series",
        "CT",
        "Cursor value: 142",
    ] {
        assert!(
            texts.iter().any(|text| text.contains(expected)),
            "overlay shows {expected:?}: {texts:?}"
        );
    }
}

#[test]
fn native_overlay_text_anonymized_mode_hides_patient_text() {
    let mut anonymized = clinical();
    anonymized.anonymize = true;
    anonymized.cursor_value = None;
    let texts = overlay_commands(&anonymized);
    for private in ["Test Patient", "ID-1", "Test Series", "20240101"] {
        assert!(
            texts.iter().all(|text| !text.contains(private)),
            "anonymized overlay hides {private:?}: {texts:?}"
        );
    }
    assert!(
        texts.iter().any(|text| text.contains("Anonymous")),
        "anonymized overlay marks the blanked corner: {texts:?}"
    );
    assert!(
        texts.iter().any(|text| text.contains("CT")),
        "anonymized overlay keeps the modality: {texts:?}"
    );
}

#[test]
fn native_overlay_text_shows_cursor_value() {
    let texts = overlay_commands(&clinical());
    assert!(
        texts.iter().any(|text| text.contains("Cursor value: 142")),
        "overlay reads out the cursor value: {texts:?}"
    );
    let mut no_cursor = clinical();
    no_cursor.cursor_value = None;
    let texts = overlay_commands(&no_cursor);
    assert!(
        texts.iter().all(|text| !text.contains("Cursor value")),
        "overlay omits the readout without a cursor: {texts:?}"
    );
}

#[test]
fn native_overlay_text_toggle_flips_show_overlay() {
    let (mut session, _root) = session();
    assert!(session.app.show_overlay);
    session
        .apply_window_action(WindowAction::ToggleOverlay)
        .expect("toggle the clinical overlay");
    assert!(!session.app.show_overlay);
    session
        .apply_window_action(WindowAction::ToggleOverlay)
        .expect("restore the clinical overlay");
    assert!(session.app.show_overlay);
}

#[test]
fn native_overlay_text_toggle_changes_composed_chrome() {
    let (mut session, _root) = session();
    session.refresh_frame().expect("compose with overlay");
    let with_overlay = session.framebuffer.pixels().to_vec();
    session
        .apply_window_action(WindowAction::ToggleOverlay)
        .expect("hide the clinical overlay");
    session.refresh_frame().expect("compose without overlay");
    let without_overlay = session.framebuffer.pixels().to_vec();
    assert_ne!(
        with_overlay, without_overlay,
        "hiding the overlay changes the composed frame"
    );
}

#[test]
fn native_overlay_text_anonymized_toggle_flips_anonymize_overlay() {
    let (mut session, _root) = session();
    assert!(!session.anonymize_overlay);
    session
        .apply_window_action(WindowAction::ToggleAnonymizedOverlay)
        .expect("toggle anonymized overlay");
    assert!(session.anonymize_overlay);
}

#[test]
fn native_overlay_text_commands_carry_single_letter_edge_labels() {
    let (session, _root) = session();
    let clinical = clinical();
    let overlay = application_overlay(
        &session.views,
        &session.viewports,
        false,
        30.0,
        Some(&clinical),
        &[],
        None,
    )
    .expect("compose the native overlay commands");
    let commands = overlay.into_display_list();
    let edge_letters: Vec<&str> = overlay_texts(commands.commands())
        .into_iter()
        .filter(|text| text.len() == 1)
        .collect();
    assert_eq!(
        edge_letters.len(),
        12,
        "four edge letters on each of three panes"
    );
    for letter in edge_letters {
        assert!(
            ["R", "L", "A", "P", "S", "I"].contains(&letter),
            "edge label {letter:?} is an orientation letter"
        );
    }
}
