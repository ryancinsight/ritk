//! A button whose release was lost or cancelled stops holding what it pressed,
//! so the next press starts over instead of finishing a stale gesture.

use super::held_gestures::{assert_nothing_held, down, key, show_two_panels, up, ARROW_RIGHT};
use super::*;
use crate::presentation::native_session::routing::{EventRoute, PanelEventRouter, PointerCapture};
use crate::presentation::{PointerButton, PresentationEvent};

#[test]
fn browsing_during_a_series_card_drag_abandons_the_drop() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    viewer.active_panel = 1;
    let (card_x, card_y) = series_card_center(&viewer, 3);
    let (x, y) = viewer.viewports[1].center();

    viewer
        .handle_events(&[down(card_x, card_y, MouseButton::Left), key(ARROW_RIGHT)])
        .expect("browse while a series card is held");
    let browsed = viewer.compare_panels[0].series_index;
    assert_eq!(browsed, Some(2), "the key must browse to the next series");

    viewer
        .handle_events(&[up(x, y, MouseButton::Left)])
        .expect("the release has no drop left to perform");
    assert_eq!(viewer.compare_panels[0].series_index, browsed);
    assert_nothing_held(&viewer);
}

#[test]
fn a_lost_card_release_leaves_nothing_for_the_next_click_to_drop() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    let (card_x, card_y) = series_card_center(&viewer, 3);
    let (x, y) = viewer.viewports[1].center();
    let shown = viewer.compare_panels[0].series_index;

    viewer
        .handle_events(&[down(card_x, card_y, MouseButton::Left)])
        .expect("press a series card");
    viewer
        .handle_events(&[down(x, y, MouseButton::Left), up(x, y, MouseButton::Left)])
        .expect("the click over a pane follows a lost release");

    assert_eq!(viewer.compare_panels[0].series_index, shown);
    assert_nothing_held(&viewer);
}

#[test]
fn a_lost_drag_button_ends_the_drop_while_another_button_is_held() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    let (card_x, card_y) = series_card_center(&viewer, 3);
    let (x, y) = viewer.viewports[1].center();
    let shown = viewer.compare_panels[0].series_index;

    viewer
        .handle_events(&[
            down(card_x, card_y, MouseButton::Left),
            down(card_x, card_y, MouseButton::Right),
        ])
        .expect("two buttons hold the card");
    viewer
        .handle_events(&[down(x, y, MouseButton::Left), up(x, y, MouseButton::Left)])
        .expect("the click over a pane follows a lost release");

    assert_eq!(viewer.compare_panels[0].series_index, shown);
}

#[test]
fn a_press_after_a_lost_release_on_the_chrome_reaches_the_pane() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    let (card_x, card_y) = series_card_center(&viewer, 3);
    let (x, y) = viewer.viewports[0].center();

    viewer
        .handle_events(&[down(card_x, card_y, MouseButton::Right)])
        .expect("press the chrome");
    assert_eq!(viewer.app.pointer.owner, None);
    viewer
        .handle_events(&[down(x, y, MouseButton::Right)])
        .expect("the press over a pane follows a lost release");

    assert_eq!(viewer.active_view, Some(0));
    assert_eq!(
        viewer.app.pointer.owner.map(|owner| owner.button),
        Some(PointerButton::Right)
    );
}

#[test]
fn focus_loss_frees_the_pointer_the_chrome_held() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    let (card_x, card_y) = series_card_center(&viewer, 3);
    let (x, y) = viewer.viewports[0].center();

    viewer
        .handle_events(&[
            down(card_x, card_y, MouseButton::Right),
            WindowEvent::FocusLost,
        ])
        .expect("the window loses focus while the chrome holds a button");
    viewer
        .handle_events(&[down(x, y, MouseButton::Middle)])
        .expect("another button presses a pane");

    assert_eq!(
        viewer.app.pointer.owner.map(|owner| owner.button),
        Some(PointerButton::Middle)
    );
}

#[test]
fn cancelling_one_button_leaves_the_pane_the_other_holds() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    let (x, y) = viewer.viewports[0].center();
    let (x, y) = (f64::from(x), f64::from(y));
    let (other_x, other_y) = viewer.viewports[1].center();
    let (other_x, other_y) = (f64::from(other_x), f64::from(other_y));
    let viewports = viewer.viewports.clone();
    let layout = viewer.workspace_layout;
    let mut router = PanelEventRouter::new(0, PointerCapture::default(), 0);
    for button in [PointerButton::Right, PointerButton::Middle] {
        let press = PresentationEvent::PointerDown { x, y, button };
        assert_eq!(
            router.route(&press, layout, &viewports),
            EventRoute::View(0)
        );
    }
    let cancel = |button| PresentationEvent::PointerCancel {
        x: other_x,
        y: other_y,
        button,
    };

    let right = cancel(PointerButton::Right);
    assert_eq!(
        router.route(&right, layout, &viewports),
        EventRoute::View(0)
    );
    assert_eq!(
        router.route(&right, layout, &viewports),
        EventRoute::Discarded
    );

    let middle = cancel(PointerButton::Middle);
    assert_eq!(
        router.route(&middle, layout, &viewports),
        EventRoute::View(0)
    );
    assert_eq!(
        router.route(&middle, layout, &viewports),
        EventRoute::Discarded
    );
}
