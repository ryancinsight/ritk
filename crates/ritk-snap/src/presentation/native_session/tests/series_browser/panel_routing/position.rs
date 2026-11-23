use super::held_gestures::{key, show_two_panels};
use super::*;

#[test]
fn native_scroll_matches_physical_planes_and_preserves_disabled_target() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let (mut viewer, _initial) = tracing::info_span!("arrange").in_scope(session);
    let root = tempfile::tempdir().expect("create the paired study directory");
    for (uid, positions) in [
        (fixtures::SERIES_UID, [10, 12, 14]),
        (SECOND_SERIES_UID, [10, 14, 18]),
    ] {
        fixtures::write_positioned_study(
            root.path(),
            "CT",
            uid,
            &fixtures::SAMPLES,
            (Some([1.5, 0.5]), true),
            positions,
            Some("2.25.259"),
        )
        .expect("write a positioned series");
    }
    show_two_panels(&mut viewer, root.path());
    viewer.active_panel = 0;
    for (keys, expected) in [
        (vec![0x26], [0, 0]),
        (vec![0x28], [1, 0]),
        (vec![0x28], [2, 1]),
        (vec![0x74, 0x26], [1, 1]),
        (vec![0x74, 0x09, 0x26], [0, 0]),
    ] {
        tracing::info_span!("act").in_scope(|| {
            for k in keys {
                viewer
                    .handle_events(&[key(k)])
                    .expect("dispatch the native input");
            }
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(
                [
                    viewer.app.axis_slice_info(0).0,
                    viewer.compare_panels[0].app.axis_slice_info(0).0
                ],
                expected
            )
        });
    }
}

#[test]
fn native_manual_alignment_preserves_selected_slices_in_different_frames() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let (mut viewer, _initial) = tracing::info_span!("arrange").in_scope(session);
    let root = tempfile::tempdir().expect("study root");
    for (uid, positions, frame) in [
        (fixtures::SERIES_UID, [10, 12, 14], "2.25.259"),
        (SECOND_SERIES_UID, [100, 102, 104], "2.25.260"),
    ] {
        fixtures::write_positioned_study(
            root.path(),
            "CT",
            uid,
            &fixtures::SAMPLES,
            (Some([1.5, 0.5]), true),
            positions,
            Some(frame),
        )
        .expect("write separate patient space");
    }
    show_two_panels(&mut viewer, root.path());
    viewer.active_panel = 0;
    viewer
        .handle_events(&[key(0x28)])
        .expect("select a different anatomical anchor");
    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x74,
            repeated: false,
            modifiers: ModifierState::SHIFT,
        }])
        .expect("align these slices");
    assert_eq!(viewer.compare_panels[0].app.axis_slice_info(0).0, 1);
    tracing::info_span!("act").in_scope(|| {
        viewer
            .handle_events(&[key(0x26)])
            .expect("scroll two millimetres")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            [
                viewer.app.axis_slice_info(0).0,
                viewer.compare_panels[0].app.axis_slice_info(0).0
            ],
            [1, 0]
        )
    });
}

#[test]
fn native_cross_reference_marks_the_active_plane_in_the_other_image() {
    use crate::presentation::native_session::layout::screen_coordinate;
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let (mut viewer, _initial) = tracing::info_span!("arrange").in_scope(session);
    let root = tempfile::tempdir().expect("study root");
    fixtures::write_positioned_study(
        root.path(),
        "CT",
        fixtures::SERIES_UID,
        &fixtures::SAMPLES,
        (Some([1.5, 0.5]), true),
        [10, 12, 14],
        Some("2.25.259"),
    )
    .expect("write sagittal source");
    fixtures::write_oriented_study(
        root.path(),
        "CT",
        SECOND_SERIES_UID,
        &fixtures::SAMPLES,
        (Some([1.5, 2.0]), true),
        fixtures::PositionedSlices {
            origins: [[10, 20, 29], [10, 20, 30], [10, 20, 31]],
            orientation: "1\\0\\0\\0\\1\\0",
            frame: Some("2.25.259"),
        },
    )
    .expect("write axial comparison");
    show_two_panels(&mut viewer, root.path());
    viewer.active_panel = 0;
    viewer.refresh_frame().expect("present the reference");
    let image = viewer.viewports[1].image;
    let at = |fraction| {
        screen_coordinate(image.x + fraction * image.width, "column").expect("screen column")
    };
    let y = screen_coordinate(image.y + 0.4 * image.height, "row").expect("screen row");
    let yellow = metis_platform::Color::rgba(255, 255, 0, 255);
    assert_eq!(viewer.framebuffer.get_pixel(at(0.375), y), yellow);
    tracing::info_span!("act")
        .in_scope(|| viewer.handle_events(&[key(0x28)]).expect("scroll source"));
    tracing::info_span!("assert").in_scope(|| {
        assert_ne!(viewer.framebuffer.get_pixel(at(0.375), y), yellow);
        assert_eq!(viewer.framebuffer.get_pixel(at(0.625), y), yellow);
    });
    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x7b,
            repeated: false,
            modifiers: ModifierState::CONTROL,
        }])
        .expect("hide lines");
    assert_ne!(viewer.framebuffer.get_pixel(at(0.625), y), yellow);
}
