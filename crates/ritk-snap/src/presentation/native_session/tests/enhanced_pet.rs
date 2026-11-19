//! Enhanced PET concatenation through the Windows native viewer.

use super::*;
use aequitas::systems::si::units::GramPerCubicCentimeter;
use tracing_subscriber::fmt::format::FmtSpan;

#[test]
fn native_enhanced_pet_concatenation_captures_and_selects_another_gate() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let root = tempfile::tempdir().expect("PET study directory");
    let (selected, capture, mut session) = tracing::info_span!("arrange").in_scope(|| {
        let [_, selected] = ritk_io::write_dynamic_gated_pet_concatenation(root.path());
        let capture = root.path().join("workstation.png");
        let mut app = SnapApp::default();
        prepare_initial_study(&mut app, &selected, None, true).expect("open concatenation");
        let session = NativeViewerSession::new_with_browser(
            app,
            Arc::new(NativeViewerObservation::default()),
            false,
            NativePresentationSelection::Fixed(NativePresentationMode::Orthogonal),
            false,
            None,
            None,
        )
        .expect("create native session");
        (selected, capture, session)
    });
    let (captured, changed) = tracing::info_span!("act").in_scope(|| {
        let captured = run_native_viewer(
            &selected,
            None,
            Some(&capture),
            NativePresentationMode::Orthogonal,
            true,
        )
        .expect("capture on the Métis Windows host");
        let changed = session
            .apply_window_action(WindowAction::StepEnhancedPetVolume(3))
            .expect("select the last temporal gate");
        (captured, changed)
    });
    let _assert = tracing::info_span!("assert").entered();
    let image = image::open(capture)
        .expect("read native window capture")
        .to_rgb8();
    assert_eq!(image.dimensions(), (INITIAL_WIDTH, INITIAL_HEIGHT));
    assert!(image.pixels().any(|pixel| pixel.0 != [0; 3]));
    assert!(captured.snapshot().loaded());
    assert_eq!(captured.snapshot().slice_counts(), [2, 1, 1]);
    assert!(changed);
    let selected = session.app.enhanced_pet_selection().expect("PET selector");
    assert_eq!(selected.selected_index(), 3);
    let volume = &selected.volumes()[3];
    assert_eq!(volume.data.as_slice(), &[2.0, 3.0]);
    assert_eq!(
        volume
            .suv_bw
            .as_ref()
            .unwrap()
            .at(1)
            .unwrap()
            .in_unit::<GramPerCubicCentimeter>(),
        7.0
    );
}
