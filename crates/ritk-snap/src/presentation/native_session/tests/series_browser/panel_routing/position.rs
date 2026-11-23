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
