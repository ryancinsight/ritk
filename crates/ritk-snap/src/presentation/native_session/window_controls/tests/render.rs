use super::*;

fn render_status(status: &str) -> Vec<u32> {
    let mut app = SnapApp::default();
    app.status_message = status.to_owned();
    render_chrome(&app)
}

fn render_chrome(app: &SnapApp) -> Vec<u32> {
    let layout = ChromeLayout::new(1_280, 800, None, app, true, WorkspaceLayout::Orthogonal)
        .expect("native chrome layout");
    let mut framebuffer =
        metis_platform::Framebuffer::new(1_280, 800).expect("native chrome framebuffer");
    layout
        .render(
            &mut framebuffer,
            app,
            &[],
            None,
            WorkspaceLayout::Orthogonal,
            0,
            &[],
        )
        .expect("render native chrome");
    framebuffer.pixels().to_vec()
}

fn packed_color(color: metis_platform::Color) -> u32 {
    u32::from_be_bytes([color.a, color.r, color.g, color.b])
}

#[test]
fn status_bar_distinguishes_a_failed_series_replacement() {
    let failure = render_status(
        "Selected series could not be opened; current panels remain: malformed DICOM",
    );
    let success = render_status("Loaded MR series (94 images).");

    assert_ne!(failure, success);
}

#[test]
fn toolbar_uses_dark_clinical_chrome_and_highlights_the_active_tool() {
    let mut app = SnapApp::default();
    app.active_tool = crate::tools::kind::ToolKind::WindowLevel;
    let pixels = render_chrome(&app);

    assert_eq!(
        pixels[2 * 1_280 + 1_270],
        packed_color(metis_platform::Color::rgb(35, 40, 47))
    );
    assert_eq!(
        pixels[31 * 1_280 + 1_270],
        packed_color(metis_platform::Color::rgb(43, 49, 57))
    );
    let layout = ChromeLayout::new(1_280, 800, None, &app, true, WorkspaceLayout::Orthogonal)
        .expect("native chrome layout");
    assert_eq!(layout.toolbar_separator_count(), 4);
    let (separator_x, separator_y) = layout
        .toolbar_separator_sample(0)
        .expect("navigation separator");
    let separator_x = usize::try_from(separator_x).expect("separator x");
    let separator_y = usize::try_from(separator_y).expect("separator y");
    assert_eq!(
        pixels[separator_y * 1_280 + separator_x],
        packed_color(metis_platform::Color::rgb(79, 88, 98))
    );
    let (x, y) = layout
        .action_center(WindowAction::SelectTool(
            crate::tools::kind::ToolKind::WindowLevel,
        ))
        .expect("window-level control position");
    let pixel_index =
        usize::try_from(y).expect("toolbar y") * 1_280 + usize::try_from(x).expect("toolbar x");
    assert_eq!(
        pixels[pixel_index],
        packed_color(metis_platform::Color::rgb(32, 105, 145))
    );
}
