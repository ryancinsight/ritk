//! Native oblique panel behavior against the real RITK event and render path.

use super::*;
use crate::app::SnapApp;
use crate::dicom::loader::tests::fixtures;
use crate::launch::NativePresentationSelection;
use crate::presentation::ViewportPoint;
use crate::tools::interaction::Annotation;
use crate::tools::ToolKind;
use crate::LoadedVolume;
use metis_platform::native::MouseButton;
use std::sync::Arc;

fn oblique_session() -> NativeViewerSession {
    let shape = [12, 10, 14];
    let [depth, rows, columns] = shape;
    let data = (0..depth)
        .flat_map(|z| {
            (0..rows).flat_map(move |y| {
                (0..columns).map(move |x| {
                    f32::from(u16::try_from(z * 101 + y * 17 + x * 3).expect("fixture value"))
                })
            })
        })
        .collect();
    let volume = LoadedVolume {
        data: Arc::new(data),
        shape,
        channels: 1,
        spacing: [2.0, 1.5, 0.8],
        origin: [0.0, 0.0, 0.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        metadata: None,
        source: None,
        modality: None,
        patient_name: None,
        patient_id: None,
        study_date: None,
        series_description: None,
        series_time: None,
        patient_weight_kg: None,
        injected_dose_bq: None,
        radionuclide_half_life_s: None,
        radiopharmaceutical_start_time: None,
        decay_correction: None,
    };
    let mut app = SnapApp::default();
    app.load_volume(volume, "analytic oblique fixture".to_owned());
    NativeViewerSession::new_with_selection(
        app,
        Arc::new(super::super::observation::NativeViewerObservation::default()),
        false,
        NativePresentationSelection::Oblique,
        false,
        None,
    )
    .expect("oblique session over an anisotropic scalar volume")
}
fn screen_point(session: &NativeViewerSession, pixel: [f64; 2]) -> (i32, i32) {
    let [x, y] = session
        .oblique_viewport
        .expect("oblique layout supplies its current mapper")
        .screen_point(pixel)
        .expect("test pixel lies inside the oblique plane");
    (
        super::super::layout::screen_coordinate(x, "test oblique pointer x")
            .expect("test pointer x is in native range"),
        super::super::layout::screen_coordinate(y, "test oblique pointer y")
            .expect("test pointer y is in native range"),
    )
}

fn click(session: &mut NativeViewerSession, x: i32, y: i32) {
    session
        .handle_events(&[WindowEvent::PointerDown {
            x,
            y,
            button: MouseButton::Left,
        }])
        .expect("oblique pointer press");
    session
        .handle_events(&[WindowEvent::PointerUp {
            x,
            y,
            button: MouseButton::Left,
        }])
        .expect("oblique pointer release");
}
mod gestures;
mod measurement;
mod render;
mod startup;
