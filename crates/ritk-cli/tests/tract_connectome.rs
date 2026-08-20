//! End-to-end exercise of `ritk tract connectome`.
//!
//! The command's parts are unit-tested inside the binary; what this covers is
//! the seam between them and the outside world — that the label volume and the
//! tractogram written by real format writers are read back into agreeing
//! physical frames, and that the matrix on disk describes the streamlines that
//! went in. Every one of those is a wiring question that a unit test on either
//! side of the boundary cannot answer.

use std::fs::File;
use std::path::Path;
use std::process::Command;

use gaia::Polyline;
use leto::geometry::Point3;

/// The command under test, as Cargo built it for this integration test.
fn ritk() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ritk"))
}

/// A 1-D strip of eight 1 mm voxels:
///
/// ```text
/// index:  0  1  2  3  4  5  6  7
/// label:  1  1  0  0  0  0  2  2
/// ```
///
/// The background gap is the white matter a streamline runs through; the
/// labelled ends are the parcels it should be attributed to.
fn write_labels(path: &Path) {
    ritk_nifti::write_nifti_labels(
        path,
        &[1, 1, 0, 0, 0, 0, 2, 2],
        // NIfTI shape is innermost-last, so the eight voxels lie along x.
        [1, 1, 8],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .expect("writing the label volume");
}

fn line(from: [f64; 3], to: [f64; 3]) -> Polyline<f64> {
    Polyline::new(vec![
        Point3::new(from[0], from[1], from[2]),
        Point3::new(to[0], to[1], to[2]),
    ])
    .expect("valid polyline")
}

fn write_tracks(path: &Path, streamlines: Vec<Polyline<f64>>) {
    let tractogram = ritk_tck::TckTractogram {
        header: ritk_tck::TckHeader::default(),
        streamlines,
    };
    let mut file = File::create(path).expect("creating the tractogram");
    tractogram.write(&mut file).expect("writing the tractogram");
}

/// Two streamlines spanning the strip end to end, one stopping a voxel short of
/// each parcel, and one staying inside the first parcel.
fn tractogram() -> Vec<Polyline<f64>> {
    vec![
        line([0.0, 0.0, 0.0], [7.0, 0.0, 0.0]),
        line([0.0, 0.0, 0.0], [7.0, 0.0, 0.0]),
        line([2.0, 0.0, 0.0], [5.0, 0.0, 0.0]),
        line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ]
}

/// The end-to-end path: two real files in, a matrix and its measures out, with
/// the weights matching the streamlines that produced them.
#[test]
fn the_command_builds_a_matrix_from_files_on_disk() {
    let directory = tempfile::tempdir().expect("temp dir");
    let labels = directory.path().join("labels.nii");
    let tracks = directory.path().join("tracks.tck");
    let matrix = directory.path().join("matrix.json");
    let measures = directory.path().join("measures.json");

    write_labels(&labels);
    write_tracks(&tracks, tractogram());

    let output = ritk()
        .args(["tract", "connectome"])
        .arg("--tractogram")
        .arg(&tracks)
        .arg("--labels")
        .arg(&labels)
        .arg("--output")
        .arg(&matrix)
        .arg("--measures")
        .arg(&measures)
        .args(["--assignment-radius", "2"])
        .output()
        .expect("running the command");

    assert!(
        output.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let encoded = std::fs::read_to_string(&matrix).expect("reading the matrix");
    let decoded: serde_json::Value = serde_json::from_str(&encoded).expect("valid JSON");

    // Two regions, and the three inter-parcel streamlines — the two spanning
    // ones plus the short one the 2 mm radius recovers — on the single edge
    // between them. The fourth stays inside one parcel and is not an edge.
    let weights = decoded["weights"].as_array().expect("weights array");
    assert_eq!(weights.len(), 4, "a two-region matrix is 2 × 2");
    let off_diagonal = weights[1].as_f64().expect("numeric weight");
    assert!(
        (off_diagonal - 3.0).abs() < 1.0e-9,
        "three streamlines connect the two parcels, got {off_diagonal}"
    );
    assert!(
        (weights[2].as_f64().expect("numeric weight") - off_diagonal).abs() < 1.0e-9,
        "the matrix must be symmetric"
    );

    let accounting = &decoded["accounting"];
    assert_eq!(accounting["total"], 4);
    assert_eq!(accounting["assigned"], 3);
    assert_eq!(accounting["intra_region"], 1);
    assert_eq!(accounting["unassigned"], 0);

    assert!(
        measures.exists(),
        "the measures file must be written when requested"
    );
    let measured: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&measures).expect("reading measures"))
            .expect("valid JSON");
    assert_eq!(measured["node_count"], 2);
    assert_eq!(measured["edge_count"], 1);
}

/// The radius is the argument that decides how much of a tractogram survives,
/// so its effect must be visible from the command line and not only in the
/// library.
#[test]
fn a_zero_radius_drops_the_streamline_that_stops_short() {
    let directory = tempfile::tempdir().expect("temp dir");
    let labels = directory.path().join("labels.nii");
    let tracks = directory.path().join("tracks.tck");
    let matrix = directory.path().join("matrix.json");

    write_labels(&labels);
    write_tracks(&tracks, tractogram());

    let output = ritk()
        .args(["tract", "connectome"])
        .arg("--tractogram")
        .arg(&tracks)
        .arg("--labels")
        .arg(&labels)
        .arg("--output")
        .arg(&matrix)
        .args(["--assignment-radius", "0"])
        .output()
        .expect("running the command");
    assert!(
        output.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let decoded: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&matrix).expect("reading the matrix"))
            .expect("valid JSON");
    let accounting = &decoded["accounting"];
    assert_eq!(
        accounting["assigned"], 2,
        "terminal assignment keeps only the two spanning streamlines"
    );
    assert_eq!(
        accounting["unassigned"], 1,
        "the streamline stopping in the gap is dropped"
    );
}

/// A tractogram format the command cannot read must say so rather than
/// producing an empty connectome, which would read as a subject with no
/// connections.
#[test]
fn an_unreadable_tractogram_format_fails_loudly() {
    let directory = tempfile::tempdir().expect("temp dir");
    let labels = directory.path().join("labels.nii");
    let tracks = directory.path().join("tracks.trx");
    let matrix = directory.path().join("matrix.json");

    write_labels(&labels);
    File::create(&tracks).expect("create");

    let output = ritk()
        .args(["tract", "connectome"])
        .arg("--tractogram")
        .arg(&tracks)
        .arg("--labels")
        .arg(&labels)
        .arg("--output")
        .arg(&matrix)
        .output()
        .expect("running the command");

    assert!(!output.status.success(), "the command must fail");
    let message = String::from_utf8_lossy(&output.stderr);
    assert!(
        message.contains("expected .tck or .trk"),
        "the error must name the formats it can read: {message}"
    );
    assert!(!matrix.exists(), "no matrix must be written on failure");
}
