use super::*;

#[test]
fn native_oblique_panels_match_public_mri_dir_source_pixels() {
    let dicom_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/2_head_mri_t2/DICOM");
    let volume = load_volume_from_path(&dicom_directory).expect("decode public MRI-DIR phantom");
    assert_eq!(volume.shape, [94, 512, 512]);
    assert_eq!(volume.spacing, [2.5, 0.5, 0.5]);

    let mut app = SnapApp::default();
    app.load_volume(volume, "public MRI-DIR phantom".to_owned());
    app.viewer_state.window_center = Some(709.0);
    app.viewer_state.window_width = Some(1_418.0);
    let session = NativeViewerSession::new_with_selection(
        app,
        std::sync::Arc::new(super::super::super::observation::NativeViewerObservation::default()),
        false,
        NativePresentationSelection::Oblique,
        true,
        None,
    )
    .expect("render public MRI-DIR phantom in all four native panes");

    let oblique = session.oblique.as_ref().expect("physical oblique view");
    assert_eq!(oblique.orientation.yaw_degrees(), 25.0);
    assert_eq!(oblique.orientation.pitch_degrees(), 15.0);
    assert_eq!(
        oblique.plane.expect("validated plane").dimensions(),
        [503, 503]
    );

    // DICOM samples use linear windowing at centre 709 and width 1418. Screen
    // coordinates avoid text, crosshairs, and pane borders in the 1280x800 view.
    for ((x, y), gray) in [
        ((300, 180), 137),
        ((360, 250), 187),
        ((950, 180), 148),
        ((1_000, 260), 157),
        ((300, 550), 114),
        ((340, 680), 124),
        ((940, 580), 141),
        ((1_000, 660), 131),
    ] {
        assert_eq!(
            session.framebuffer.get_pixel(x, y),
            metis_platform::Color::rgba(gray, gray, gray, 255),
            "decoded DICOM sample at screen pixel ({x}, {y})"
        );
    }
}

#[test]
fn invalid_oblique_source_geometry_keeps_the_last_rendered_frame() {
    let mut session = oblique_session();
    let original_framebuffer = session.framebuffer.pixels().to_vec();
    let original_viewport = session.oblique_viewport;
    let oblique = session.oblique.as_ref().expect("oblique view");
    let original_pixels = oblique.frame.rgba().to_vec();
    let original_plane = oblique.plane.expect("validated plane");
    let original_orientation = oblique.orientation;

    session.app.loaded.as_mut().expect("loaded source").spacing[0] = f64::NAN;
    session.app.bump_visual_revision();
    let error = session
        .oblique
        .as_mut()
        .expect("oblique view")
        .render_if_stale(&mut session.app)
        .expect_err("non-finite source spacing cannot replace a valid frame");

    assert!(format!("{error:#}").contains("positive and finite"));
    assert_eq!(session.framebuffer.pixels(), original_framebuffer);
    assert_eq!(session.oblique_viewport, original_viewport);
    let oblique = session.oblique.as_ref().expect("retained oblique view");
    assert_eq!(oblique.frame.rgba(), original_pixels);
    assert_eq!(oblique.plane, Some(original_plane));
    assert_eq!(oblique.orientation, original_orientation);
}

#[test]
fn native_oblique_panel_renders_a_real_fourth_reslice_and_labels_it() {
    let session = oblique_session();
    let oblique = session.oblique.as_ref().expect("oblique selection renders");
    let viewport = session.oblique_viewport.expect("oblique mapper");
    let plane = oblique.plane.expect("source geometry builds a plane");
    viewport
        .validate_dimensions(plane.dimensions())
        .expect("render and pointer mapper share dimensions");
    assert_eq!(
        session.presentation_mode,
        NativePresentationSelection::Oblique
    );
    assert!(
        oblique
            .frame
            .rgba()
            .chunks_exact(4)
            .any(|pixel| pixel[..3].iter().any(|channel| *channel != 0))
    );
    assert!(
        session
            .framebuffer
            .pixels()
            .iter()
            .any(|pixel| *pixel != 0xFF00_0000)
    );

    let overlay = super::super::layout::surface_frames_with_oblique(
        &session.views,
        &oblique.frame,
        oblique.orientation.yaw_degrees(),
        oblique.orientation.pitch_degrees(),
        INITIAL_WIDTH,
        INITIAL_HEIGHT,
        session.app.zoom,
        session.app.pan_offset,
        oblique.zoom,
        oblique.pan_offset,
        Some(&plane),
        session.app.cine.enabled,
        session.app.cine.fps,
        true,
    )
    .expect("oblique application layout");
    assert_eq!(overlay.0.width(), INITIAL_WIDTH);
    assert_eq!(overlay.0.height(), INITIAL_HEIGHT);
    let (x, y) = screen_point(
        &session,
        [
            f64::from(u32::try_from(plane.dimensions()[0] - 1).expect("width")) * 0.5,
            f64::from(u32::try_from(plane.dimensions()[1] - 1).expect("height")) * 0.5,
        ],
    );
    assert_ne!(
        session.framebuffer.get_pixel(x, y),
        metis_platform::Color::BLACK,
        "the fourth panel presents sampled grayscale anatomy"
    );
}
