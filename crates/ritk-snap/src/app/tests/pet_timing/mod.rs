//! Classic PET timing through Part 10 loading, pointer mapping, and fusion.

mod cases;

use crate::app::viewer_viewport::ViewerViewport;
use crate::app::SnapApp;
use crate::dicom::load_dicom_volume;
use crate::presentation::{ViewerAction, ViewportPoint};
use crate::ui::ViewTransform;

use self::cases::timing_cases;

#[test]
fn classic_pet_suv_preserves_verified_timing_and_withholds_unavailable_values() {
    let _subscriber = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let directory = tempfile::tempdir().expect("classic PET fixture directory");
    let mut failures = Vec::new();
    for (index, case) in timing_cases().into_iter().enumerate() {
        let (mut app, viewport) =
            tracing::info_span!("arrange", case.name, ?case.image.decay_correction).in_scope(
                || {
                    let path = directory.path().join(index.to_string());
                    std::fs::create_dir(&path).expect("isolated PET series directory");
                    case.image.write(&path.join("first.dcm"), 1, 0);
                    if let Some(second) = case.second {
                        second.write(&path.join("second.dcm"), 2, 2);
                    }
                    let volume = load_dicom_volume(&path)
                        .expect("load classic PET through the viewer loader");
                    let app = SnapApp {
                        loaded: Some(volume),
                        ..SnapApp::default()
                    };
                    let viewport = ViewerViewport::new(
                        0,
                        [0.0; 2],
                        [1.0; 2],
                        [2, 2],
                        ViewTransform::default(),
                    )
                    .expect("fixture viewport");
                    (app, viewport)
                },
            );
        let (samples, fusion) = tracing::info_span!("act", case.name).in_scope(|| {
            let samples = [0.25, 1.25].map(|column| {
                app.apply_viewer_action(
                    &ViewerAction::PointerMoved {
                        position: ViewportPoint::new(column, 0.25),
                    },
                    Some(&viewport),
                )
                .expect("pointer action reaches the loaded voxel");
                (app.pointer_intensity, app.pointer_suv)
            });
            #[cfg(feature = "eframe-shell")]
            let fusion = Some(fusion_matches(
                app.loaded.as_ref().expect("loaded PET for fusion"),
                case.suv.unwrap_or(case.first_pixel),
            ));
            #[cfg(not(feature = "eframe-shell"))]
            let fusion: Option<bool> = None;
            (samples, fusion)
        });
        tracing::info_span!("assert", case.name).in_scope(|| {
            assert_eq!(
                samples.map(|(raw, _)| raw),
                [case.first_pixel, case.first_pixel * 1.5]
            );
            let expected = [case.suv, case.suv.map(|suv| suv * 1.5)];
            let matches = samples
                .into_iter()
                .zip(expected)
                .all(|((_, observed), expected)| {
                    match (observed, expected) {
                        // The independent decimal oracle is rounded to the
                        // pointer's precision. Allow two steps for rounding the
                        // analytical value and the 1.5 multiple separately.
                        (Some(actual), Some(reference)) => {
                            actual.to_bits().abs_diff(reference.to_bits()) <= 2
                        }
                        (None, None) => true,
                        _ => false,
                    }
                });
            if !matches || fusion == Some(false) {
                failures.push(format!(
                    "{} {:?}: observed {samples:?}, expected {expected:?}, fusion {fusion:?}",
                    case.name, case.image.decay_correction
                ));
            }
        });
    }
    assert!(
        failures.is_empty(),
        "{} PET timing cases failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[cfg(feature = "eframe-shell")]
fn fusion_matches(volume: &crate::LoadedVolume, expected: f32) -> bool {
    use crate::render::{
        render_fused_slice, FusedSliceParams, GrayscalePresentation, NamedColorMap, WindowLevel,
    };

    let window = WindowLevel::new(1.5, 3.0);
    let source = || FusedSliceParams {
        volume,
        axis: 0,
        slice: 0,
        wl: window,
        colormap: NamedColorMap::Grayscale,
    };
    let fused = render_fused_slice(source(), source(), 1.0).expect("aligned PET fusion");
    let presentation = GrayscalePresentation::for_volume(volume).expect("fixture presentation");
    fused.pixels[..2]
        == [expected, expected * 1.5]
            .map(|value| egui::Color32::from_gray(presentation.apply(window, f64::from(value))))
}
