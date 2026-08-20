//! End-to-end exercise of `ritk parcellate atlas`.
//!
//! The command's parts are unit-tested inside the binary. What this covers is
//! the seam between them and the outside world: that intensity and label
//! volumes written by a real format writer come back as a subject and its
//! atlases, that the labels the registration warps land on the subject's grid
//! in the file the caller named, and that a disagreement between atlases is
//! resolved by the fusion rule and reported in the agreement map. Every one of
//! those is a wiring question no unit test on either side of the boundary can
//! answer.

use std::path::{Path, PathBuf};
use std::process::Command;

use coeus_core::SequentialBackend;

/// The command under test, as Cargo built it for this integration test.
fn ritk() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ritk"))
}

/// Shape of the test volume, outermost axis first.
///
/// Deliberately unequal on every axis, and paired with anisotropic spacing
/// below. A cubic volume on an isotropic grid cannot fail an axis-order test:
/// reverse the axes and every array is the same length and every voxel the
/// same size, so the defect cancels itself and the test passes while the
/// geometry is wrong.
const SHAPE: [usize; 3] = [10, 12, 14];
/// Voxel size along each axis, in millimetres — unequal for the same reason.
const SPACING: [f32; 3] = [2.0, 1.5, 1.0];
/// Number of voxels in a test volume.
const VOXELS: usize = SHAPE[0] * SHAPE[1] * SHAPE[2];
/// Half-open index range of the foreground block, per axis. Offset differently
/// on each axis so that a permutation moves it.
const BLOCK: [std::ops::Range<usize>; 3] = [2..5, 4..8, 7..12];
/// Foreground intensity. Any value clearly above the background will do; the
/// registration metric is scale-invariant.
const FOREGROUND: u32 = 100;

/// Row-major index of a voxel in the test volume.
fn at(i: usize, j: usize, k: usize) -> usize {
    (i * SHAPE[1] + j) * SHAPE[2] + k
}

/// A block of `value` on a background of zero.
///
/// Both the intensity images and the label volumes are this shape, which is
/// what makes the expected parcellation knowable: the labels the atlas carries
/// cover exactly the structure the registration is matching.
fn block(value: u32) -> Vec<u32> {
    let mut volume = vec![0_u32; VOXELS];
    for i in BLOCK[0].clone() {
        for j in BLOCK[1].clone() {
            for k in BLOCK[2].clone() {
                volume[at(i, j, k)] = value;
            }
        }
    }
    volume
}

/// Write a volume as a NIfTI on an anisotropic identity-oriented grid.
fn write(path: &Path, volume: &[u32]) {
    ritk_nifti::write_nifti_labels(
        path,
        volume,
        SHAPE,
        [0.0, 0.0, 0.0],
        SPACING,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .expect("writing the volume");
}

/// One atlas on disk: an intensity volume and the labels that go with it.
struct Atlas {
    intensity: PathBuf,
    labels: PathBuf,
}

/// Write an atlas whose intensity matches the subject exactly and whose labels
/// mark the cube with `label`.
///
/// An atlas identical to the subject is what makes the expected output exact
/// rather than approximate: there is no deformation to recover, so the warp is
/// the identity and the labels must arrive unchanged. A test that instead
/// deformed the atlas would be measuring the registration's accuracy, which
/// belongs to the registration's own tests.
fn write_atlas(directory: &Path, index: usize, label: u32) -> Atlas {
    let intensity = directory.join(format!("atlas{index}.nii"));
    let labels = directory.join(format!("atlas{index}_labels.nii"));
    write(&intensity, &block(FOREGROUND));
    write(&labels, &block(label));
    Atlas { intensity, labels }
}

/// Run `parcellate atlas` over the given atlases and return its stdout.
fn parcellate(
    subject: &Path,
    atlases: &[Atlas],
    output: &Path,
    agreement: Option<&Path>,
    extra: &[&str],
) -> String {
    let mut command = ritk();
    command
        .args(["parcellate", "atlas"])
        .arg("--subject")
        .arg(subject);
    for atlas in atlases {
        command
            .arg("--atlas-intensity")
            .arg(&atlas.intensity)
            .arg("--atlas-labels")
            .arg(&atlas.labels);
    }
    command.arg("--output").arg(output);
    if let Some(path) = agreement {
        command.arg("--agreement").arg(path);
    }
    // A short schedule keeps the test inside its budget. It is honest here
    // because the atlases already sit on the subject, so there is nothing for
    // the extra iterations to close.
    command.args(["--iterations", "4,2"]).args(extra);

    let run = command.output().expect("running the command");
    assert!(
        run.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout).into_owned()
}

/// Read a label volume back off disk.
fn read_labels(path: &Path) -> Vec<u32> {
    let (labels, shape) = ritk_nifti::read_nifti_labels(path).expect("reading the parcellation");
    assert_eq!(
        shape, SHAPE,
        "the parcellation must land on the subject's grid"
    );
    labels
}

/// Read the agreement map back off disk as the ordinary float image it is.
fn read_agreement(path: &Path) -> Vec<f32> {
    let backend = SequentialBackend::default();
    let image = ritk_nifti::read_nifti::<SequentialBackend, _>(path, &backend)
        .expect("reading the agreement map");
    image
        .data_slice()
        .expect("the agreement map is contiguous")
        .to_vec()
}

// ── The single-atlas path ────────────────────────────────────────────────

/// The whole pipeline over one atlas: files in, a parcellation on the
/// subject's grid out, carrying the labels the atlas supplied.
///
/// The assertion is exact rather than an overlap score because the atlas is
/// the subject: any voxel that changed label did so through a wiring defect —
/// a grid mismatch, an axis reversal, or a label truncated on its way through
/// the float representation — not through registration error.
#[test]
fn one_atlas_transfers_its_labels_onto_the_subject() {
    let directory = tempfile::tempdir().expect("temp dir");
    let subject = directory.path().join("subject.nii");
    let output = directory.path().join("parcellation.nii");

    write(&subject, &block(FOREGROUND));
    let atlases = [write_atlas(directory.path(), 0, 7)];

    let stdout = parcellate(&subject, &atlases, &output, None, &[]);

    assert_eq!(
        read_labels(&output),
        block(7),
        "an atlas already on the subject must transfer its labels unchanged"
    );
    assert!(
        stdout.contains("1 regions"),
        "the report must name the one region found, got: {stdout}"
    );
}

/// A single atlas has nothing to disagree with, so every voxel is unanimous.
/// The map must still be written — a caller diffing agreement across runs
/// needs the one-atlas case to be present and saturated, not absent.
#[test]
fn a_single_atlas_agrees_with_itself_everywhere() {
    let directory = tempfile::tempdir().expect("temp dir");
    let subject = directory.path().join("subject.nii");
    let output = directory.path().join("parcellation.nii");
    let agreement = directory.path().join("agreement.nii");

    write(&subject, &block(FOREGROUND));
    let atlases = [write_atlas(directory.path(), 0, 1)];

    parcellate(&subject, &atlases, &output, Some(&agreement), &[]);

    let map = read_agreement(&agreement);
    assert_eq!(map.len(), VOXELS);
    for (index, value) in map.iter().enumerate() {
        assert!(
            (value - 1.0).abs() < 1.0e-6,
            "voxel {index} reads {value}, but one atlas is unanimous by construction"
        );
    }
}

// ── Fusion across several atlases ────────────────────────────────────────

/// Two atlases against one is the case fusion exists for: the majority must
/// win, and the agreement map must say the vote was not unanimous.
///
/// Two thirds is the value that distinguishes a real vote from a map that
/// merely reports whether a label was found — the failure mode a
/// presence-only check would pass.
#[test]
fn the_majority_label_wins_and_the_dissent_is_recorded() {
    let directory = tempfile::tempdir().expect("temp dir");
    let subject = directory.path().join("subject.nii");
    let output = directory.path().join("parcellation.nii");
    let agreement = directory.path().join("agreement.nii");

    write(&subject, &block(FOREGROUND));
    let atlases = [
        write_atlas(directory.path(), 0, 1),
        write_atlas(directory.path(), 1, 1),
        write_atlas(directory.path(), 2, 2),
    ];

    parcellate(&subject, &atlases, &output, Some(&agreement), &[]);

    assert_eq!(
        read_labels(&output),
        block(1),
        "two atlases voting for label 1 must outvote the one voting for 2"
    );

    let map = read_agreement(&agreement);
    for i in BLOCK[0].clone() {
        for j in BLOCK[1].clone() {
            for k in BLOCK[2].clone() {
                let value = map[at(i, j, k)];
                assert!(
                    (value - 2.0 / 3.0).abs() < 1.0e-5,
                    "voxel ({i},{j},{k}) reads {value}; two of three atlases agreed there"
                );
            }
        }
    }
    // Outside the cube every atlas labelled background, so the vote is
    // unanimous — which is what proves the two-thirds above is a measured
    // share and not a constant the writer emits everywhere.
    assert!(
        (map[at(0, 0, 0)] - 1.0).abs() < 1.0e-6,
        "the background is unanimous, got {}",
        map[at(0, 0, 0)]
    );
}

/// Joint label fusion weights each atlas by how well it matches locally. With
/// interchangeable atlases it must reach the same answer as majority voting —
/// a different one would mean the weighting is reading something other than
/// the local match.
#[test]
fn joint_fusion_agrees_with_the_majority_when_the_atlases_are_interchangeable() {
    let directory = tempfile::tempdir().expect("temp dir");
    let subject = directory.path().join("subject.nii");
    let output = directory.path().join("parcellation.nii");

    write(&subject, &block(FOREGROUND));
    let atlases = [
        write_atlas(directory.path(), 0, 1),
        write_atlas(directory.path(), 1, 1),
        write_atlas(directory.path(), 2, 2),
    ];

    parcellate(&subject, &atlases, &output, None, &["--fusion", "joint"]);

    assert_eq!(read_labels(&output), block(1));
}

// ── Argument handling ────────────────────────────────────────────────────

/// Unpaired atlas arguments must fail before any registration runs. Pairing
/// them positionally is convenient but silent: a caller who omits one label
/// volume would otherwise get a parcellation built from atlases whose labels
/// belong to different brains.
#[test]
fn an_atlas_without_its_labels_is_rejected() {
    let directory = tempfile::tempdir().expect("temp dir");
    let subject = directory.path().join("subject.nii");
    let output = directory.path().join("parcellation.nii");

    write(&subject, &block(FOREGROUND));
    let first = write_atlas(directory.path(), 0, 1);
    let second = write_atlas(directory.path(), 1, 2);

    let run = ritk()
        .args(["parcellate", "atlas"])
        .arg("--subject")
        .arg(&subject)
        .arg("--atlas-intensity")
        .arg(&first.intensity)
        .arg("--atlas-intensity")
        .arg(&second.intensity)
        .arg("--atlas-labels")
        .arg(&first.labels)
        .arg("--output")
        .arg(&output)
        .output()
        .expect("running the command");

    assert!(!run.status.success(), "unpaired atlases must not succeed");
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("2 intensities and 1 label volumes"),
        "the error must name the mismatch, got: {stderr}"
    );
    assert!(
        !output.exists(),
        "nothing may be written when the arguments are rejected"
    );
}

/// An atlas on a different grid cannot be fused, and saying so is the whole
/// point: a registration recovers a deformation, never a resampling, so a
/// mismatched atlas silently accepted would produce labels for a brain of a
/// different size.
#[test]
fn an_atlas_off_the_subject_grid_is_rejected_with_both_sizes() {
    let directory = tempfile::tempdir().expect("temp dir");
    let subject = directory.path().join("subject.nii");
    let intensity = directory.path().join("small.nii");
    let labels = directory.path().join("small_labels.nii");
    let output = directory.path().join("parcellation.nii");

    write(&subject, &block(FOREGROUND));
    let small = vec![0_u32; 8];
    for path in [&intensity, &labels] {
        ritk_nifti::write_nifti_labels(
            path,
            &small,
            [2, 2, 2],
            [0.0, 0.0, 0.0],
            SPACING,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .expect("writing the small atlas");
    }

    let run = ritk()
        .args(["parcellate", "atlas"])
        .arg("--subject")
        .arg(&subject)
        .arg("--atlas-intensity")
        .arg(&intensity)
        .arg("--atlas-labels")
        .arg(&labels)
        .arg("--output")
        .arg(&output)
        .output()
        .expect("running the command");

    assert!(!run.status.success(), "a mismatched grid must not succeed");
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains(&VOXELS.to_string()) && stderr.contains('8'),
        "the error must name both sizes, got: {stderr}"
    );
}
