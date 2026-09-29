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
    let success = rendered_status("Loaded MR series (94 instances).");

    assert_ne!(failure, success);
}
