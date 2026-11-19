//! Native fit and pixel-scale display controls.

use super::*;

#[test]
fn fit_key_restores_unit_zoom_and_repaints() {
    let (mut session, _root) = session();
    session.app.zoom = 2.5;
    session.app.pan_offset = crate::tools::interaction::ViewportOffset::new(3.0, -7.0);
    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: VIRTUAL_KEY_ZOOM_FIT,
            repeated: false,
            modifiers: ModifierState::CONTROL,
        }])
        .expect("fit key");
    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert_eq!(session.app.zoom, 1.0);
    assert_eq!(
        session.app.pan_offset,
        crate::tools::interaction::ViewportOffset::new(0.0, 0.0)
    );
}

#[test]
fn one_to_one_action_maps_normalized_texels_after_rotation() {
    let (mut session, _root) = session();
    assert!(session
        .apply_window_action(WindowAction::ZoomOneToOne)
        .expect("zoom 1:1"));
    session.refresh_frame().expect("recompose at 1:1 zoom");
    let frame = session.views[0].frame();
    assert_eq!([frame.width(), frame.height()], [4, 2]);
    assert_eq!(frame.display_spacing().values(), [1.5, 0.5]);
    let viewport = session.viewports[0];
    let actual = [
        viewport.image.width / f64::from(frame.width()),
        viewport.image.height / f64::from(frame.height()),
    ];
    let expected = [1.0 / 3.0, 1.0];
    // The target zoom narrows once to f32. Two f32 epsilons cover that
    // rounding and the subsequent f64 placement products and quotients.
    let relative_bound = 2.0 * f64::from(f32::EPSILON);
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((actual - expected).abs() <= relative_bound * expected.abs().max(1.0));
    }

    assert!(session
        .apply_window_action(WindowAction::RotateCw)
        .expect("rotate clockwise"));
    session.refresh_frame().expect("recompose rotated frame");
    assert!(session
        .apply_window_action(WindowAction::ZoomOneToOne)
        .expect("zoom rotated frame 1:1"));
    session
        .refresh_frame()
        .expect("recompose rotated frame at 1:1 zoom");
    let frame = session.views[0].frame();
    assert_eq!([frame.width(), frame.height()], [2, 4]);
    assert_eq!(frame.display_spacing().values(), [0.5, 1.5]);
    let viewport = session.viewports[0];
    let actual = [
        viewport.image.width / f64::from(frame.width()),
        viewport.image.height / f64::from(frame.height()),
    ];
    let expected = [1.0, 1.0 / 3.0];
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((actual - expected).abs() <= relative_bound * expected.abs().max(1.0));
    }
}
