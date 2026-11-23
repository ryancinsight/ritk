//! Reader journey on the downloaded TCIA MRI-DIR porcine phantom.
use super::*;
use crate::presentation::native_session::tests::support::control_center;
use crate::presentation::native_session::window_controls::Menu;
use metis_platform::native::MouseButton;

fn menu_click(viewer: &mut NativeViewerSession, menu: Option<Menu>, action: WindowAction) {
    let (x, y) = control_center(viewer, menu, action);
    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x,
                y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x,
                y,
                button: MouseButton::Left,
            },
        ])
        .expect("use the visible menu row");
}

/// Capture the presented region of each image. Fractional fit scales assign a
/// screen pixel to the half-open image rectangle by its center (x + ½, y + ½).
fn displayed_images(viewer: &NativeViewerSession) -> Vec<Vec<u32>> {
    let stride = usize::try_from(viewer.framebuffer.width()).expect("framebuffer width");
    let rows = viewer.framebuffer.pixels().chunks_exact(stride);
    assert_eq!(viewer.viewports.len(), 3);
    viewer
        .viewports
        .iter()
        .map(|pane| {
            let mut image = Vec::new();
            for (y, row) in rows.clone().enumerate() {
                let cy = f64::from(u32::try_from(y).expect("surface row")) + 0.5;
                for (x, &pixel) in row.iter().enumerate() {
                    let cx = f64::from(u32::try_from(x).expect("surface column")) + 0.5;
                    if pane.contains(cx, cy) && pane.image.contains(cx, cy) {
                        image.push(pixel);
                    }
                }
            }
            assert!(
                image.windows(2).any(|pair| pair[0] != pair[1]),
                "expected visible MRI detail"
            );
            image
        })
        .collect()
}

#[test]
fn native_anonymize_reopens_real_mri_with_identical_voxels_and_three_rendered_planes() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/2_head_mri_t2/DICOM");
    let (mut viewer, _initial) = tracing::info_span!("arrange").in_scope(session);
    viewer
        .open_study_path(&source)
        .expect("open the downloaded 94-instance phantom");
    viewer
        .apply_window_action(WindowAction::ToggleOverlay)
        .expect("hide labels for the image comparison");
    assert!(!viewer.app.show_overlay);
    viewer.refresh_frame().expect("display source planes");
    let displayed = displayed_images(&viewer);
    let volume = viewer.app.loaded.as_ref().expect("read source");
    let values = std::sync::Arc::clone(&volume.data);
    assert_eq!(volume.shape, [94, 512, 512]);
    assert_ne!(volume.patient_id.as_deref(), Some("ANON001"));
    let planes = viewer.views.each_ref().map(|v| v.frame.rgba().to_vec());
    let parent = tempfile::tempdir().expect("local destination");
    let output = parent.path().join("strict");
    tracing::info_span!("act").in_scope(|| {
        viewer
            .anonymize_to(
                &output,
                AnonymizationScope::Series,
                Deidentification::Strict,
            )
            .expect("write verified copy");
        menu_click(&mut viewer, None, WindowAction::OpenMenu(Menu::Anonymize));
        menu_click(
            &mut viewer,
            Some(Menu::Anonymize),
            WindowAction::OpenExportedCopy,
        );
    });
    let _check = tracing::info_span!("assert").entered();
    assert_eq!(viewer.study_root.as_deref(), Some(output.as_path()));
    let reopened = viewer.app.loaded.as_ref().expect("decoded copy");
    assert_eq!(reopened.patient_name.as_deref(), Some("ANONYMOUS"));
    assert_eq!(reopened.patient_id.as_deref(), Some("ANON001"));
    assert_eq!(reopened.shape, [94, 512, 512]);
    assert!(values
        .iter()
        .zip(reopened.data.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert_eq!(values.len(), reopened.data.len());
    assert!(
        displayed_images(&viewer) == displayed,
        "reopened MRI changed the presented image pixels"
    );
    for (view, expected) in viewer.views.iter().zip(planes) {
        assert!(
            view.frame.rgba() == expected,
            "export changed pixels on axis {}",
            view.axis
        );
    }
}

#[test]
fn native_anonymize_failed_export_or_new_study_clears_reopen_target() {
    let (mut viewer, root) = session_on_study();
    let parent = tempfile::tempdir().expect("output parent");
    let output = parent.path().join("first");
    viewer
        .anonymize_to(
            &output,
            AnonymizationScope::Series,
            Deidentification::Strict,
        )
        .expect("first export");
    assert!(viewer
        .anonymize_to(
            &output,
            AnonymizationScope::Series,
            Deidentification::Strict
        )
        .is_err());
    viewer
        .apply_window_action(WindowAction::OpenExportedCopy)
        .expect("no completed export");
    assert_eq!(viewer.study_root.as_deref(), Some(root.path()));
    assert!(viewer.app.status_message.contains("Export a DICOM copy"));
    let second = parent.path().join("second");
    viewer
        .anonymize_to(&second, AnonymizationScope::Study, Deidentification::Basic)
        .expect("second export");
    viewer
        .open_study_path(&output)
        .expect("explicitly open another folder");
    viewer
        .apply_window_action(WindowAction::OpenExportedCopy)
        .expect("new study has no export");
    assert_eq!(viewer.study_root.as_deref(), Some(output.as_path()));
}
