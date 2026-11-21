//! Quantitative ROI coverage through the host-neutral pointer adapter.

use super::{apply_app_event, ViewerViewport};
use crate::app::tests::test_volume;
use crate::app::SnapApp;
use crate::presentation::{
    AnnotationKind, PointerButton, PresentationEvent, PresentationModifiers,
};
use crate::tools::interaction::{Annotation, AnnotationLocation, RoiKind, ToolState};
use crate::tools::kind::ToolKind;
use crate::ui::tool_shortcuts::{VIRTUAL_KEY_ROI_ELLIPSE, VIRTUAL_KEY_ROI_RECT};
use crate::ui::ViewTransform;
use std::sync::Arc;
use tracing_subscriber::fmt::format::FmtSpan;

const SIZE: [usize; 2] = [7, 6];
const SCALE: f64 = 16.0;
const LOCATION: Option<AnnotationLocation> = Some(AnnotationLocation { plane: 0, slice: 1 });

fn with_trace(test: &'static str, run: impl FnOnce()) {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    tracing::info_span!("roi_quantitative", test).in_scope(run);
}

fn loaded() -> SnapApp {
    let mut volume = test_volume([2, 6, 7]);
    // Separate slices by 1000 HU; a wrong slice cannot look like a correct ROI.
    // Within each slice I(row,col) = 10*row + col, in HU.
    volume.data = Arc::new(
        (0_u8..2)
            .flat_map(|slice| {
                (0_u8..6).flat_map(move |row| {
                    (0_u8..7).map(move |col| {
                        1000.0 * f32::from(slice) + 10.0 * f32::from(row) + f32::from(col)
                    })
                })
            })
            .collect(),
    );
    volume.spacing = [3.0, 2.0, 0.75];
    let mut app = SnapApp::default();
    app.load_volume(volume, "ROI calibration fixture".to_owned());
    app.set_slice_for_axis(0, 1);
    assert_eq!(app.axis_slice_info(0), (1, 2));
    app
}

fn view(origin: [f64; 2]) -> ViewerViewport {
    ViewerViewport::new(0, origin, [SCALE; 2], SIZE, ViewTransform::default())
        .expect("positive, finite viewport")
}

fn host_point(origin: [f64; 2], [column, row]: [f64; 2]) -> [f64; 2] {
    [origin[0] + SCALE * column, origin[1] + SCALE * row]
}

fn select(app: &mut SnapApp, view: &ViewerViewport, virtual_key: u32, kind: ToolKind) {
    apply_app_event(
        app,
        view,
        PresentationEvent::KeyDown {
            virtual_key,
            repeated: false,
            modifiers: PresentationModifiers::NONE,
        },
    );
    assert_eq!(app.active_tool, kind);
}

fn begin_drag(
    app: &mut SnapApp,
    view: &ViewerViewport,
    [x, y]: [f64; 2],
    [end_x, end_y]: [f64; 2],
) {
    for event in [
        PresentationEvent::PointerDown {
            x,
            y,
            button: PointerButton::Left,
            modifiers: PresentationModifiers::NONE,
        },
        PresentationEvent::PointerMove { x: end_x, y: end_y },
    ] {
        apply_app_event(app, view, event);
    }
}

fn release(app: &mut SnapApp, view: &ViewerViewport, [x, y]: [f64; 2]) {
    apply_app_event(
        app,
        view,
        PresentationEvent::PointerUp {
            x,
            y,
            button: PointerButton::Left,
        },
    );
}

fn assert_pointer_released(app: &SnapApp) {
    assert!(app.tool_state.is_idle());
    assert!(app.pointer.owner.is_none());
    assert_eq!(app.pointer.drag_measurement_location, None);
}

fn assert_rounding(actual: f32, expected: f64) {
    // The small integer sums in these fixtures are exact in f32. Allow
    // rounding of division and square root (SD) or the f32 value of pi and
    // multiplications (area), not any whole-pixel or physical-axis change.
    let bound = 4.0 * f64::from(f32::EPSILON) * expected.abs();
    assert!(
        (f64::from(actual) - expected).abs() <= bound,
        "expected {expected} within {bound}, got {actual}"
    );
}

fn assert_summary(app: &SnapApp, kind: AnnotationKind, area_mm2: f32) {
    let snap = app.presentation_snapshot();
    assert_eq!(snap.annotation_count(), 1);
    let summary = snap.last_annotation().expect("completed ROI is reported");
    assert_eq!(summary.kind(), kind);
    assert_eq!(summary.primary_value(), f64::from(area_mm2));
}

#[test]
fn rectangle_pointer_roi_values_are_invariant_at_equal_image_points() {
    with_trace("rectangle area under translated hosts", || {
        // Equal fractional image coordinates under two fractional host origins;
        // the inclusive sample bounds are rows 1..=3 and columns 1..=4.
        for origin in [[16.0, 496.453_125], [16.0, 302.953_125]] {
            let _case =
                tracing::info_span!("viewport_origin", x = origin[0], y = origin[1]).entered();
            let (view, mut app, from, to) = tracing::info_span!("arrange").in_scope(|| {
                // Reverse the drag to verify normalized opposite corners.
                (
                    view(origin),
                    loaded(),
                    host_point(origin, [3.75, 2.75]),
                    host_point(origin, [1.25, 1.25]),
                )
            });
            tracing::info_span!("act").in_scope(|| {
                select(&mut app, &view, VIRTUAL_KEY_ROI_RECT, ToolKind::RoiRect);
                begin_drag(&mut app, &view, from, to);
                release(&mut app, &view, to);
            });
            let _assert = tracing::info_span!("assert").entered();
            assert_pointer_released(&app);
            let [Annotation::RoiRect {
                location,
                top_left,
                bottom_right,
                mean,
                std_dev,
                min,
                max,
                area_mm2,
            }] = app.annotations.as_slice()
            else {
                panic!("one completed rectangle is required");
            };
            assert_eq!(*location, LOCATION);
            assert_eq!(*top_left, [1.25, 1.25]);
            assert_eq!(*bottom_right, [2.75, 3.75]);
            assert_eq!((*mean, *min, *max), (1022.5, 1011.0, 1034.0));
            // Independent variance = 10² * var(1,2,3) + var(1,2,3,4).
            assert_rounding(*std_dev, (815.0_f64 / 12.0).sqrt());
            assert_eq!(*area_mm2, 18.0); // 3*2.0 mm × 4*0.75 mm.
            assert_summary(&app, AnnotationKind::RoiRect, *area_mm2);
        }
    });
}

#[test]
fn ellipse_pointer_roi_values_follow_anisotropic_physical_semiaxes() {
    with_trace("ellipse area on anisotropic pixels", || {
        let (view, mut app, from, to) = tracing::info_span!("arrange").in_scope(|| {
            let origin = [4.0, 80.0];
            (
                view(origin),
                loaded(),
                host_point(origin, [1.0, 1.0]),
                host_point(origin, [5.0, 5.0]),
            )
        });
        tracing::info_span!("act").in_scope(|| {
            select(
                &mut app,
                &view,
                VIRTUAL_KEY_ROI_ELLIPSE,
                ToolKind::RoiEllipse,
            );
            begin_drag(&mut app, &view, from, to);
            release(&mut app, &view, to);
        });
        let _assert = tracing::info_span!("assert").entered();
        assert_pointer_released(&app);
        let [Annotation::RoiEllipse {
            location,
            center,
            radii,
            mean,
            std_dev,
            min,
            max,
            area_mm2,
        }] = app.annotations.as_slice()
        else {
            panic!("one completed ellipse is required");
        };
        assert_eq!(*location, LOCATION);
        assert_eq!(*center, [3.0, 3.0]);
        assert_eq!(*radii, [2.0, 2.0]);
        // 13 samples with rows: {3}, {2,3,4}, {1,2,3,4,5}, {2,3,4}, {3}.
        // Squared row/column offsets each sum to 14; the cross term cancels.
        assert_eq!((*mean, *min, *max), (1033.0, 1013.0, 1053.0));
        assert_rounding(*std_dev, (1414.0_f64 / 13.0).sqrt());
        assert_rounding(*area_mm2, 6.0 * std::f64::consts::PI); // π*4 mm*1.5 mm.
        assert_summary(&app, AnnotationKind::RoiEllipse, *area_mm2);
    });
}

#[test]
fn cancelling_either_roi_never_reports_a_measurement() {
    with_trace("cancel completed pointer motion", || {
        for (key, tool, roi) in [
            (VIRTUAL_KEY_ROI_RECT, ToolKind::RoiRect, RoiKind::Rect),
            (
                VIRTUAL_KEY_ROI_ELLIPSE,
                ToolKind::RoiEllipse,
                RoiKind::Ellipse,
            ),
        ] {
            let _case = tracing::info_span!("roi_kind", ?roi).entered();
            let (view, mut app, from, to) = tracing::info_span!("arrange").in_scope(|| {
                let origin = [4.0, 80.0];
                (
                    view(origin),
                    loaded(),
                    host_point(origin, [1.0, 1.0]),
                    host_point(origin, [5.0, 5.0]),
                )
            });
            tracing::info_span!("act", phase = "drag").in_scope(|| {
                select(&mut app, &view, key, tool);
                begin_drag(&mut app, &view, from, to);
            });
            tracing::info_span!("assert", phase = "before_cancel").in_scope(|| {
                assert_eq!(
                    app.pointer.owner.map(|owner| owner.button),
                    Some(PointerButton::Left)
                );
                assert_eq!(app.pointer.drag_measurement_location, LOCATION);
                assert!(
                    matches!(
                        &app.tool_state,
                        ToolState::RoiDrag { start, current, kind }
                            if *kind == roi && *start != *current
                    ),
                    "expected moved {roi:?} before cancellation, got {:?}",
                    app.tool_state
                );
            });
            tracing::info_span!("act", phase = "cancel").in_scope(|| {
                apply_app_event(
                    &mut app,
                    &view,
                    PresentationEvent::PointerCancel {
                        x: to[0],
                        y: to[1],
                        button: PointerButton::Left,
                    },
                );
            });
            let _assert = tracing::info_span!("assert", phase = "after_cancel").entered();
            assert_pointer_released(&app);
            assert!(app.annotations.is_empty());
            assert_eq!(app.presentation_snapshot().annotation_count(), 0);
            assert_eq!(app.presentation_snapshot().last_annotation(), None);
        }
    });
}
