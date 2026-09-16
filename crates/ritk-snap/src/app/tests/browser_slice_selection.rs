//! Exact browser slice-selection state transitions and rejection cases.

use super::*;
use crate::app::browser_slice_selection::{
    parse_browser_slice_request, BrowserSliceCoordinate, BrowserSliceSelectionError,
};
use crate::ui::LinkedCursor;

#[test]
fn exact_selection_updates_each_axis_and_linked_cursor() {
    let shape = [3, 4, 5];
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume(shape));
    app.linked_cursor = Some(LinkedCursor::from_slices(shape, 0, 0, 0));

    assert_eq!(app.select_browser_slice(0, 2), Ok(true));
    assert_eq!(app.select_browser_slice(1, 3), Ok(true));
    assert_eq!(app.select_browser_slice(2, 4), Ok(true));

    assert_eq!(app.axis_slice_info(0), (2, 3));
    assert_eq!(app.axis_slice_info(1), (3, 4));
    assert_eq!(app.axis_slice_info(2), (4, 5));
    assert_eq!(app.linked_cursor.expect("linked cursor").voxel(), [2, 3, 4]);
    assert_eq!(app.visual_revision, 3);
}

#[test]
fn selecting_current_slice_requires_no_repaint() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([3, 4, 5]));
    app.coronal_slice = 2;
    let revision = app.visual_revision;

    assert_eq!(app.select_browser_slice(1, 2), Ok(false));
    assert_eq!(app.coronal_slice, 2);
    assert_eq!(app.visual_revision, revision);
}

#[test]
fn selection_rejects_unloaded_and_invalid_coordinates_without_mutation() {
    let mut unloaded = SnapApp::default();
    assert_eq!(
        unloaded.select_browser_slice(0, 0),
        Err(BrowserSliceSelectionError::StudyNotLoaded)
    );

    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([3, 4, 5]));
    app.viewer_state.slice_index = 1;
    app.coronal_slice = 2;
    app.sagittal_slice = 3;
    let revision = app.visual_revision;

    assert_eq!(
        app.select_browser_slice(3, 0),
        Err(BrowserSliceSelectionError::AxisOutOfRange { axis: 3 })
    );
    assert_eq!(
        app.select_browser_slice(1, 4),
        Err(BrowserSliceSelectionError::IndexOutOfRange {
            axis: 1,
            index: 4,
            slice_count: 4,
        })
    );
    assert_eq!(
        (
            app.viewer_state.slice_index,
            app.coronal_slice,
            app.sagittal_slice,
            app.visual_revision,
        ),
        (1, 2, 3, revision)
    );
}

#[test]
fn zero_length_axis_rejects_every_index() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([3, 0, 5]));

    assert_eq!(
        app.select_browser_slice(1, 0),
        Err(BrowserSliceSelectionError::IndexOutOfRange {
            axis: 1,
            index: 0,
            slice_count: 0,
        })
    );
}

#[test]
fn wasm_numbers_convert_only_after_exact_integer_validation() {
    assert_eq!(parse_browser_slice_request(2.0, 4.0), Ok((2, 4)));
    assert_eq!(
        parse_browser_slice_request(-0.0, f64::from(u32::MAX)),
        Ok((
            0,
            usize::try_from(u32::MAX).expect("invariant: test target represents u32")
        ))
    );
}

#[test]
fn wasm_numbers_reject_non_finite_fractional_and_wrapping_values() {
    assert!(matches!(
        parse_browser_slice_request(f64::NAN, 0.0),
        Err(BrowserSliceSelectionError::CoordinateNotFinite {
            coordinate: BrowserSliceCoordinate::Axis,
            value,
        }) if value.is_nan()
    ));
    assert_eq!(
        parse_browser_slice_request(0.0, f64::INFINITY),
        Err(BrowserSliceSelectionError::CoordinateNotFinite {
            coordinate: BrowserSliceCoordinate::Index,
            value: f64::INFINITY,
        })
    );
    assert_eq!(
        parse_browser_slice_request(1.5, 0.0),
        Err(BrowserSliceSelectionError::CoordinateNotInteger {
            coordinate: BrowserSliceCoordinate::Axis,
            value: 1.5,
        })
    );
    assert_eq!(
        parse_browser_slice_request(0.0, -1.0),
        Err(BrowserSliceSelectionError::CoordinateOutOfRange {
            coordinate: BrowserSliceCoordinate::Index,
            value: -1.0,
            maximum: u32::MAX,
        })
    );
    let wraps_to_zero_in_wasm = f64::from(u32::MAX) + 1.0;
    assert_eq!(
        parse_browser_slice_request(wraps_to_zero_in_wasm, wraps_to_zero_in_wasm),
        Err(BrowserSliceSelectionError::CoordinateOutOfRange {
            coordinate: BrowserSliceCoordinate::Axis,
            value: wraps_to_zero_in_wasm,
            maximum: u32::MAX,
        })
    );
}
