use super::*;

use leto::geometry::Point3;
use ritk_parcellation::ParcellationGrid;

/// A 1-D strip of eight 1 mm voxels with a labelled parcel at each end and
/// background between them.
fn strip() -> Parcellation {
    let grid =
        ParcellationGrid::axis_aligned([8, 1, 1], [1.0, 1.0, 1.0], [0.0; 3]).expect("valid grid");
    Parcellation::new(Box::new([1, 1, 0, 0, 0, 0, 2, 2]), grid, Vec::new())
        .expect("valid parcellation")
}

fn line(from: [f64; 3], to: [f64; 3]) -> Polyline<f64> {
    Polyline::new(vec![
        Point3::new(from[0], from[1], from[2]),
        Point3::new(to[0], to[1], to[2]),
    ])
    .expect("valid polyline")
}

// ── Weighting maps to the library's own enum ─────────────────────────────

/// Every command-line choice must reach the library variant it names. A
/// mismatch here would silently build a different matrix from the one the
/// caller asked for, which no downstream check catches because every variant
/// produces a valid matrix.
#[test]
fn every_weighting_choice_maps_to_its_library_variant() {
    assert_eq!(
        EdgeWeighting::from(Weighting::Count),
        EdgeWeighting::StreamlineCount
    );
    assert_eq!(
        EdgeWeighting::from(Weighting::InverseLength),
        EdgeWeighting::InverseLength
    );
    assert_eq!(
        EdgeWeighting::from(Weighting::InverseNodeVolume),
        EdgeWeighting::InverseNodeVolume
    );
    assert_eq!(
        EdgeWeighting::from(Weighting::MeanLength),
        EdgeWeighting::MeanLength
    );
}

// ── The radius argument selects the assignment ───────────────────────────

/// A zero radius must select terminal assignment rather than a zero-radius
/// search, and a positive one the radial search. The two differ in how much of
/// a tractogram survives, so the boundary is worth pinning.
#[test]
fn the_radius_argument_selects_the_assignment() {
    let parcellation = strip();
    // A streamline stopping one voxel short of each parcel.
    let short = [line([2.0, 0.0, 0.0], [5.0, 0.0, 0.0])];

    let terminal = ConnectomeConfig::new().with_assignment(EndpointAssignment::Terminal);
    let radial = ConnectomeConfig::new()
        .with_assignment(EndpointAssignment::RadialSearch { radius_mm: 2.0 });

    let dropped = build_connectivity_matrix(&parcellation, &short, &terminal).expect("build");
    let recovered = build_connectivity_matrix(&parcellation, &short, &radial).expect("build");

    assert_eq!(dropped.accounting().unassigned, 1);
    assert_eq!(recovered.accounting().assigned, 1);
}

// ── Tractogram reading ───────────────────────────────────────────────────

#[test]
fn an_unrecognised_extension_is_rejected() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("tracks.vtk");
    File::create(&path).expect("create");

    let error = read_streamlines(&path).expect_err("the rejected input must yield the typed error");
    assert!(
        error.to_string().contains("expected .tck or .trk"),
        "got {error}"
    );
}

#[test]
fn a_path_without_an_extension_is_rejected() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("tracks");
    File::create(&path).expect("create");

    let error = read_streamlines(&path).expect_err("the rejected input must yield the typed error");
    assert!(
        error.to_string().contains("cannot infer a track format"),
        "got {error}"
    );
}

/// Streamlines written as `.tck` must come back as the same polylines, since
/// every endpoint lookup depends on their coordinates surviving the round trip.
#[test]
fn tck_streamlines_round_trip_through_the_reader() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("tracks.tck");

    let written = vec![
        line([0.0, 0.0, 0.0], [7.0, 0.0, 0.0]),
        line([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    ];
    let tractogram = ritk_tck::TckTractogram {
        header: ritk_tck::TckHeader::default(),
        streamlines: written.clone(),
    };
    let mut file = File::create(&path).expect("create");
    tractogram.write(&mut file).expect("write");
    file.flush().expect("flush");
    drop(file);

    let read = read_streamlines(&path).expect("read");
    assert_eq!(read.len(), written.len());
    for (index, (left, right)) in read.iter().zip(&written).enumerate() {
        assert_eq!(left.len(), right.len(), "streamline {index}");
        for (point, expected) in left.points().iter().zip(right.points()) {
            assert!(
                (point.x - expected.x).abs() < 1.0e-4
                    && (point.y - expected.y).abs() < 1.0e-4
                    && (point.z - expected.z).abs() < 1.0e-4,
                "streamline {index}: {point:?} vs {expected:?}"
            );
        }
    }
}

// ── Label volumes ────────────────────────────────────────────────────────

/// A label volume arrives as floats, so the reader has to recover integers from
/// them. Negative and zero values are background, and a positive value rounds
/// to its nearest label rather than truncating toward it — truncation would
/// turn every value that the format stored a hair under its integer into the
/// label below.
#[test]
fn label_values_recover_from_their_float_representation() {
    let voxels: [f32; 6] = [-1.0, 0.0, 0.4, 1.0, 16.999_998, 17.000_002];
    let recovered: Vec<u32> = voxels
        .iter()
        .map(|value| {
            if *value <= 0.0 {
                0
            } else {
                value.round() as u32
            }
        })
        .collect();
    assert_eq!(recovered, vec![0, 0, 0, 1, 17, 17]);
}
