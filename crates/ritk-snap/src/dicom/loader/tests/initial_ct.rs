//! Opt-in acceptance on the owner's local CT. Paths and source identifiers stay out of diagnostics.
use super::super::{initial::first_readable, load_volume_from_series_info, scan_folder_for_series};
use crate::presentation::PresentationFrame;
use crate::render::{NamedColorMap, WindowLevel};
use std::path::{Path, PathBuf};

fn read_initial(path: &Path) -> crate::LoadedVolume {
    let tree = scan_folder_for_series(path).expect("scan local five-series CT");
    let acquisitions: Vec<_> = tree
        .iter_series()
        .map(|series| &series.acquisition)
        .collect();
    let counts: Vec<_> = acquisitions.iter().map(|a| a.file_paths.len()).collect();
    assert_eq!(counts, [2, 34, 67, 111, 89]);
    let (index, volume) = first_readable(acquisitions.len(), |i| {
        load_volume_from_series_info(acquisitions[i])
    })
    .expect("open first readable CT series from disk");
    assert_eq!(index, 1);
    assert_eq!(volume.shape, [34, 512, 512]);
    volume
}

#[test]
#[ignore = "requires the retained CT source and its local Basic and Strict export folders"]
fn retained_ct_native_initial_series_and_planes_match_both_exports() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let _arrange = tracing::info_span!("arrange").entered();
    let source = PathBuf::from(std::env::var_os("RITK_RETAINED_CT").expect("RITK_RETAINED_CT"));
    let copies = PathBuf::from(std::env::var_os("RITK_CT_EXPORTS").expect("RITK_CT_EXPORTS"));
    let original = read_initial(&source);
    drop(_arrange);
    for profile in ["Basic", "Strict"] {
        let copy =
            tracing::info_span!("act", profile).in_scope(|| read_initial(&copies.join(profile)));
        let _check = tracing::info_span!("assert", profile).entered();
        assert!(
            original.data.as_slice() == copy.data.as_slice(),
            "initial CT voxels changed"
        );
        assert_eq!(original.spacing, copy.spacing);
        assert_eq!(original.origin, copy.origin);
        assert_eq!(original.direction, copy.direction);
        for axis in 0..3 {
            let render = |volume: &crate::LoadedVolume| {
                PresentationFrame::from_slice(
                    volume,
                    axis,
                    volume.shape[axis] / 2,
                    WindowLevel::new(40., 400.),
                    NamedColorMap::Grayscale,
                )
                .expect("render middle CT plane")
            };
            assert!(
                render(&original).rgba() == render(&copy).rgba(),
                "initial CT plane changed"
            );
        }
    }
}

#[test]
#[ignore = "requires the owner's local five-series CT"]
fn retained_ct_native_reformatted_positions_and_intersections() {
    use crate::geometry::patient_stack::PatientStack;
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let source = PathBuf::from(std::env::var_os("RITK_RETAINED_CT").expect("RITK_RETAINED_CT"));
    let tree = tracing::info_span!("arrange")
        .in_scope(|| scan_folder_for_series(&source).expect("scan CT"));
    let series: Vec<_> = tree.iter_series().map(|s| &s.acquisition).collect();
    assert_eq!(
        series
            .iter()
            .map(|s| s.file_paths.len())
            .collect::<Vec<_>>(),
        [2, 34, 67, 111, 89]
    );
    let [thick, thin] = tracing::info_span!("act").in_scope(|| {
        [1, 2].map(|i| {
            let volume = load_volume_from_series_info(series[i]).expect("decode CT reconstruction");
            PatientStack::displayed_planes(&volume).map(|s| s.expect("qualified display plane"))
        })
    });
    let _check = tracing::info_span!("assert").entered();
    for axis in [1, 2] {
        for i in 0..512 {
            assert_eq!(thick[axis].corresponding_slice(i, &thin[axis]), Some(i));
            assert_eq!(thin[axis].corresponding_slice(i, &thick[axis]), Some(i));
        }
        for z in 0_u16..34 {
            let trace = thick[0]
                .trace_on(usize::from(z), &thin[axis], 233)
                .expect("intersect target");
            assert_eq!(trace.distance([12., f64::from(2 * z) - 33.]), 0.);
        }
    }
}
