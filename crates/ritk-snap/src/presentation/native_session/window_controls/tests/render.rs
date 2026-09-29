use super::*;

fn rendered_status(status: &str) -> Vec<u32> {
    let mut app = SnapApp::default();
    app.status_message = status.to_owned();
    let layout = ChromeLayout::new(1_280, 800, None, &app, true, WorkspaceLayout::Orthogonal)
        .expect("native chrome layout");
    let mut framebuffer =
        metis_platform::Framebuffer::new(1_280, 800).expect("native status framebuffer");
    layout
        .render(
            &mut framebuffer,
            &app,
            &[],
            None,
            WorkspaceLayout::Orthogonal,
            0,
            &[],
        )
        .expect("render native status");
    framebuffer.pixels().to_vec()
}

#[test]
fn status_bar_distinguishes_a_failed_series_replacement() {
    let failure = rendered_status(
        "Selected series could not be opened; current panels remain: malformed DICOM",
    );
    let success = rendered_status("Loaded MR series (94 images).");

    assert_ne!(failure, success);
}

#[test]
fn toolbar_draws_all_five_group_labels_above_controls() {
    let pixels = rendered_status("Ready");
    let background = pixels[33 * 1_280 + 1_270];
    let groups = [(12, 80), (302, 380), (500, 570), (652, 730), (814, 890)];

    for (left, right) in groups {
        assert!(
            (33..43).any(|y| (left..right).any(|x| pixels[y * 1_280 + x] != background)),
            "expected a visible toolbar group label in x={left}..{right}"
        );
    }
}
