//! Series-tray rendering value tests.

use super::*;

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
