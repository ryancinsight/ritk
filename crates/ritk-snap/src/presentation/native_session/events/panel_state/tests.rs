use super::*;

#[test]
fn keyboard_series_navigation_uses_the_predicted_browser_cursor() {
    let layout = WorkspaceLayout::Panels(PanelGrid::new(2, 1).expect("two-panel grid"));
    let displayed_series = std::array::from_fn(|index| (index == 0).then_some(0));
    let mut state = KeyboardPanelState::new(layout, 0, None, 0, None, displayed_series, Some(0))
        .expect("valid panel state");

    state
        .apply_action(
            WindowAction::BrowseSeries {
                series_index: 1,
                panel_index: 0,
            },
            true,
        )
        .expect("predict first series browse");
    state
        .apply_action(WindowAction::ActivateNextPanel, true)
        .expect("predict focus change to the empty panel");

    assert_eq!(state.active_panel(), 1);
    let expected = [Some(1); MAX_GRID_PANELS];
    assert_eq!(state.navigation_series(), expected);
}

#[test]
fn closing_a_panel_resets_the_predicted_browser_cursor_to_primary() {
    let layout = WorkspaceLayout::Panels(PanelGrid::new(3, 1).expect("three-panel grid"));
    let displayed_series = std::array::from_fn(|index| match index {
        0 => Some(0),
        1 => Some(2),
        _ => None,
    });
    let mut state = KeyboardPanelState::new(layout, 1, None, 0, None, displayed_series, Some(2))
        .expect("valid panel state");

    state
        .apply_action(WindowAction::CloseActivePanel, true)
        .expect("predict panel close");

    let expected = [Some(0); MAX_GRID_PANELS];
    assert_eq!(state.navigation_series(), expected);
}
