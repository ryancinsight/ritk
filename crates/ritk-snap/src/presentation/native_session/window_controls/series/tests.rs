//! Series-tray rendering value tests.

use super::navigator::displayed_preview;
use super::thumbnails::{draw_thumbnail, fit_dimensions};
use super::*;
use arrayvec::ArrayString;

#[test]
fn series_cards_show_order_within_their_study() {
    use crate::dicom::loader::{scan_folder_for_series, tests::fixtures};

    let root = tempfile::tempdir().expect("series root");
    for (index, modality) in ["MR", "CT", "MR"].into_iter().enumerate() {
        fixtures::write_study(root.path(), modality, &format!("2.25.20260906{index:04}"))
            .expect("write series into the shared study");
    }
    let tree = scan_folder_for_series(root.path()).expect("scan the shared study");
    let browser = SeriesBrowser::from_tree(&tree, None).expect("build the series catalog");

    assert_eq!(browser.study_count(), 1);
    assert_eq!(browser.len(), 3);
    let first = browser.choice(0).expect("first series");
    assert_eq!(first.acquisition.study_date(), Some("20260905"));
    assert_eq!(
        first.acquisition.study_instance_uid(),
        Some("2.25.20260905")
    );
    assert_eq!(first.study_series_number, 1);
    assert_eq!(first.study_series_count, 3);
    let labels = (0..browser.len())
        .map(|index| {
            cards::series_position_label(browser.choice(index).expect("series choice"))
                .expect("series position")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        labels.iter().map(ArrayString::as_str).collect::<Vec<_>>(),
        ["Series 1/3", "Series 2/3", "Series 3/3"]
    );
}

#[test]
fn first_series_card_shows_separate_patient_and_study_header_boxes() {
    use crate::dicom::loader::{scan_folder_for_series, tests::fixtures};

    let root = tempfile::tempdir().expect("series root");
    fixtures::write_study(root.path(), "MR", "2.25.202609050001").expect("write series");
    let tree = scan_folder_for_series(root.path()).expect("scan series");
    let browser = SeriesBrowser::from_tree(&tree, None).expect("build series catalog");
    let area = Rect::new(0, 0, 300, 400);
    let mut framebuffer = Framebuffer::new(300, 400).expect("series rail framebuffer");

    cards::render(
        &mut framebuffer,
        area,
        Some(&browser),
        &SnapApp::default(),
        &[],
        0,
        &[],
    )
    .expect("render grouped series card");

    assert_eq!(
        framebuffer.get_pixel(24, 62),
        Color::rgb(27, 45, 59),
        "the patient details have their own preview-rail header box"
    );
    assert_eq!(
        framebuffer.get_pixel(24, 83),
        Color::rgb(38, 54, 67),
        "the study details have their own preview-rail header box"
    );
}

#[test]
fn thumbnail_preserves_sampled_rgba_from_the_rendered_slice() {
    let frame = PresentationFrame::from_rgba(2, 1, &[255, 0, 0, 255, 0, 255, 0, 255])
        .expect("two-pixel axial frame");
    let mut framebuffer = Framebuffer::new(8, 4).expect("thumbnail framebuffer");

    draw_thumbnail(&mut framebuffer, Rect::new(1, 1, 4, 2), &frame)
        .expect("render the input frame into the preview");

    assert_eq!(framebuffer.get_pixel(1, 1), Color::rgba(255, 0, 0, 255));
    assert_eq!(framebuffer.get_pixel(2, 1), Color::rgba(255, 0, 0, 255));
    assert_eq!(framebuffer.get_pixel(3, 1), Color::rgba(0, 255, 0, 255));
    assert_eq!(framebuffer.get_pixel(4, 1), Color::rgba(0, 255, 0, 255));
}

#[test]
fn each_assigned_series_preview_uses_its_own_panel_frame() {
    let primary =
        PresentationFrame::from_rgba(1, 1, &[255, 0, 0, 255]).expect("primary series thumbnail");
    let comparison =
        PresentationFrame::from_rgba(1, 1, &[0, 255, 0, 255]).expect("comparison series thumbnail");
    let previews = [Some(&primary), Some(&comparison)];

    let displayed_series = [Some(3), Some(7)];
    let primary_preview =
        displayed_preview(3, &displayed_series, &previews).expect("primary series is displayed");
    let comparison_preview =
        displayed_preview(7, &displayed_series, &previews).expect("comparison series is displayed");

    assert_eq!(primary_preview.rgba(), &[255, 0, 0, 255]);
    assert_eq!(comparison_preview.rgba(), &[0, 255, 0, 255]);
    assert_eq!(
        displayed_preview(9, &displayed_series, &previews).map(PresentationFrame::rgba),
        None
    );
}

#[test]
fn thumbnail_fit_preserves_aspect_ratio_without_cropping() {
    assert_eq!(
        fit_dimensions(512, 256, 70, 56).expect("fit wide slice"),
        (70, 35)
    );
    assert_eq!(
        fit_dimensions(256, 512, 70, 56).expect("fit tall slice"),
        (28, 56)
    );
}

#[test]
fn thumbnail_rejects_zero_source_extent() {
    assert_eq!(
        fit_dimensions(0, 256, 70, 56)
            .expect_err("zero-width DICOM slice")
            .to_string(),
        "series preview dimensions must be nonzero"
    );
}

#[test]
fn thumbnail_count_badge_renders_the_series_image_count() {
    let thumbnail = Rect::new(4, 5, 56, 72);
    let mut ninety_four = Framebuffer::new(64, 80).expect("first count badge framebuffer");
    let mut four_hundred_nine = Framebuffer::new(64, 80).expect("second count badge framebuffer");

    cards::draw_image_count(&mut ninety_four, thumbnail, 94)
        .expect("render the first series image count");
    cards::draw_image_count(&mut four_hundred_nine, thumbnail, 409)
        .expect("render the second series image count");

    assert_eq!(ninety_four.get_pixel(40, 60), Color::rgb(23, 58, 77));
    let first_pixels = (0..80)
        .flat_map(|y| (0..64).map(move |x| (x, y)))
        .map(|(x, y)| ninety_four.get_pixel(x, y))
        .collect::<Vec<_>>();
    let second_pixels = (0..80)
        .flat_map(|y| (0..64).map(move |x| (x, y)))
        .map(|(x, y)| four_hundred_nine.get_pixel(x, y))
        .collect::<Vec<_>>();

    assert_ne!(first_pixels, second_pixels);
}
