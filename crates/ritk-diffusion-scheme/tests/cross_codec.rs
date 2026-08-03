//! ADR 0036 verification condition 8 — cross-codec differential test.
//!
//! One gradient scheme is written through the FSL, MRtrix, NRRD, and DICOM
//! codecs, read back by each codec, and asserted pairwise identical for all
//! voxel-relevant metadata: direction vectors and b-values per volume.
//!
//! Frame conventions differ per codec (FSL/MRtrix → ImageAxis,
//! NRRD/DICOM → Lps), so the frame is asserted per-codec rather than
//! across.
//!
//! The multi-dataset tests use the downloaded OpenNeuro datasets from
//! `test_data/diffusion/` and are `#[ignore]` by default.

use std::path::PathBuf;

use ritk_diffusion_scheme::{
    DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme, read_fsl_scheme,
    read_mrtrix_scheme, write_fsl_scheme, write_mrtrix_scheme,
};
use ritk_spatial::Vector;

// ── Helpers ──────────────────────────────────────────────────────────────

fn weighting(value: f64) -> DiffusionWeighting {
    DiffusionWeighting::from_seconds_per_square_millimeter(value).expect("finite weighting")
}

fn make_three_volume_single_shell_scheme() -> GradientScheme {
    GradientScheme::new(
        vec![
            GradientDirection::new(weighting(0.0), Vector::new([0.0, 0.0, 0.0])).unwrap(),
            GradientDirection::new(weighting(1_000.0), Vector::new([1.0, 0.0, 0.0])).unwrap(),
            GradientDirection::new(weighting(1_000.0), Vector::new([0.0, 1.0, 0.0])).unwrap(),
        ],
        GradientFrame::Lps,
    )
    .expect("valid scheme")
}

fn assert_scheme_equivalence(expected: &GradientScheme, actual: &GradientScheme) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "scheme lengths differ"
    );
    for (i, (exp, act)) in expected
        .directions()
        .iter()
        .zip(actual.directions().iter())
        .enumerate()
    {
        assert_eq!(
            exp.weighting(),
            act.weighting(),
            "b-value at index {i} differs: expected {:?}, got {:?}",
            exp.weighting().seconds_per_square_millimeter(),
            act.weighting().seconds_per_square_millimeter(),
        );
        let expected_dir = exp.direction();
        let actual_dir = act.direction();
        assert!(
            (expected_dir.to_array()[0] - actual_dir.to_array()[0]).abs() < 1e-9
                && (expected_dir.to_array()[1] - actual_dir.to_array()[1]).abs() < 1e-9
                && (expected_dir.to_array()[2] - actual_dir.to_array()[2]).abs() < 1e-9,
            "direction at index {i} differs: expected {:?}, got {:?}",
            expected_dir.to_array(),
            actual_dir.to_array(),
        );
    }
}

/// Round-trip a scheme through all four codecs, returning FSL, MRtrix,
/// NRRD, and DICOM results in that order.
fn cross_codec_round_trip_all(scheme: &GradientScheme) -> [GradientScheme; 4] {
    [fsl_round_trip(scheme), mrtrix_round_trip(scheme), nrrd_round_trip(scheme), dicom_round_trip(scheme)]
}

// ── FSL round-trip ───────────────────────────────────────────────────────

fn fsl_round_trip(scheme: &GradientScheme) -> GradientScheme {
    let (bval, bvec) = write_fsl_scheme(scheme);
    read_fsl_scheme(&bval, &bvec).expect("FSL round-trip parse")
}

// ── MRtrix round-trip ────────────────────────────────────────────────────

fn mrtrix_round_trip(scheme: &GradientScheme) -> GradientScheme {
    let header = write_mrtrix_scheme(scheme);
    read_mrtrix_scheme(&header).expect("MRtrix round-trip parse")
}

// ── NRRD round-trip ──────────────────────────────────────────────────────

/// Write a single-shell scheme through the NRRD gradient-scheme codec.
///
/// The NRRD DWI convention uses one nominal `DWMRI_b-value` and scales
/// each volume's effective weighting by `(norm / max_norm)²`.  This
/// function derives the nominal b-value from the scheme's maximum
/// (non-b0) weighting and writes unit-length gradient vectors so the
/// effective b-value equals the nominal for every DWI volume.
fn nrrd_round_trip(scheme: &GradientScheme) -> GradientScheme {
    use std::io::Write;

    let max_b = scheme
        .directions()
        .iter()
        .map(|entry| entry.weighting().seconds_per_square_millimeter())
        .max_by(f64::total_cmp)
        .unwrap_or(1_000.0);

    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("cross_codec.nrrd");

    let count = scheme.len();
    let mut file = std::fs::File::create(&path).expect("create NRRD file");
    writeln!(file, "NRRD0005").unwrap();
    writeln!(file, "type: float").unwrap();
    writeln!(file, "dimension: 4").unwrap();
    writeln!(file, "space: left-posterior-superior").unwrap();
    writeln!(file, "sizes: {count} 2 2 2").unwrap();
    writeln!(file, "space directions: none (1,0,0) (0,1,0) (0,0,1)").unwrap();
    writeln!(file, "kinds: list domain domain domain").unwrap();
    writeln!(file, "encoding: raw").unwrap();
    writeln!(file, "modality:=DWMRI").unwrap();
    writeln!(file, "DWMRI_b-value:={max_b}").unwrap();
    for (i, entry) in scheme.directions().iter().enumerate() {
        let [x, y, z] = entry.direction().to_array();
        writeln!(file, "DWMRI_gradient_{i:04}:={x} {y} {z}").unwrap();
    }
    writeln!(file).unwrap();
    drop(file);

    ritk_nrrd::read_nrrd_gradient_scheme(path).expect("NRRD round-trip read")
}

// ── DICOM round-trip ─────────────────────────────────────────────────────

fn dicom_round_trip(scheme: &GradientScheme) -> GradientScheme {
    use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
    use dicom::object::InMemDicomObject;
    use ritk_dicom::attribute::tags;
    use std::path::Path;

    fn write_instance(
        path: &Path,
        weighting: f64,
        direction: &[f64],
        uid: &str,
    ) -> anyhow::Result<()> {
        let mut object = InMemDicomObject::new_empty();
        object.put(DataElement::new(
            Tag::from(tags::DIFFUSION_B_VALUE),
            VR::FD,
            PrimitiveValue::from(weighting),
        ));
        object.put(DataElement::new(
            Tag::from(tags::DIFFUSION_GRADIENT_DIRECTION),
            VR::FD,
            PrimitiveValue::F64(dicom::core::smallvec::SmallVec::from_vec(
                direction.to_vec(),
            )),
        ));
        let object = object.with_meta(
            dicom::object::meta::FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.4")
                .media_storage_sop_instance_uid(uid)
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )?;
        object.write_to_file(path)?;
        Ok(())
    }

    let directory = tempfile::tempdir().expect("tempdir");
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    for (i, entry) in scheme.directions().iter().enumerate() {
        let b = entry.weighting().seconds_per_square_millimeter();
        let [x, y, z] = entry.direction().to_array();
        let path = directory.path().join(format!("instance_{i:04}.dcm"));
        write_instance(&path, b, &[x, y, z], &format!("2.25.{}", 100 + i))
            .expect("write DICOM instance");
        paths.push(path);
    }

    ritk_dicom::diffusion::read_dicom_gradient_scheme_from_files(&paths)
        .expect("DICOM round-trip read")
}

// ── Cross-codec differential test (synthetic) ────────────────────────────

#[test]
fn cross_codec_all_agree_on_single_shell_scheme() {
    let original = make_three_volume_single_shell_scheme();

    let fsl = fsl_round_trip(&original);
    let mrtrix = mrtrix_round_trip(&original);
    let nrrd = nrrd_round_trip(&original);
    let dicom = dicom_round_trip(&original);

    // Per-codec frame conventions.
    assert_eq!(fsl.frame(), GradientFrame::ImageAxis);
    assert_eq!(mrtrix.frame(), GradientFrame::ImageAxis);
    assert_eq!(nrrd.frame(), GradientFrame::Lps);
    assert_eq!(dicom.frame(), GradientFrame::Lps);

    // Directions and b-values must agree regardless of frame.
    assert_scheme_equivalence(&original, &fsl);
    assert_scheme_equivalence(&original, &mrtrix);
    assert_scheme_equivalence(&original, &nrrd);
    assert_scheme_equivalence(&original, &dicom);
}

// ── Multi-dataset helpers ────────────────────────────────────────────────

/// CARGO_MANIFEST_DIR → test_data/diffusion base.
fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/diffusion")
}

fn ds002087_dwi_dir() -> PathBuf {
    test_data_dir().join("ds002087_repo/sub-01/dwi")
}

fn ds004666_dwi_dir() -> PathBuf {
    test_data_dir().join("ds004666_repo/sub-01/ses-0p9mm/dwi")
}

fn ds002087_available() -> bool {
    ds002087_dwi_dir()
        .join("sub-01_run-1_dwi.bval")
        .exists()
}

fn ds004666_available() -> bool {
    ds004666_dwi_dir()
        .join("sub-01_ses-0p9mm_dir-AP_dwi.bval")
        .exists()
}

/// Normalise gradient directions so they pass the unit-vector contract.
///
/// Real bvec files often have directions whose Euclidean norm is within
/// `1e-7` of unity — close enough for analysis but not for the `1e-6`
/// validation in [`GradientDirection::new`].
fn normalise_scheme(scheme: GradientScheme) -> Option<GradientScheme> {
    let entries = scheme
        .directions()
        .iter()
        .map(|entry| {
            let dir = entry.direction();
            let norm = dir.norm();
            if norm < 1e-15 {
                return Some(*entry);
            }
            let unit = Vector::new([
                dir.to_array()[0] / norm,
                dir.to_array()[1] / norm,
                dir.to_array()[2] / norm,
            ]);
            GradientDirection::new(entry.weighting(), unit).ok()
        })
        .collect::<Option<Vec<_>>>()?;
    GradientScheme::new(entries, scheme.frame()).ok()
}

/// Subset a scheme to a single shell, keeping b0-equivalent and `b ≥ min_b`
/// volumes.
fn subset_by_b(scheme: &GradientScheme, min_b: f64, b0_cutoff: f64) -> GradientScheme {
    let entries = scheme
        .directions()
        .iter()
        .filter(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            b <= b0_cutoff || b >= min_b
        })
        .cloned()
        .collect::<Vec<_>>();
    GradientScheme::new(entries, scheme.frame()).expect("subset must be valid")
}

/// Read a scheme from FSL bval/bvec in a data directory.
fn load_scheme_from_fsl(bval_path: &std::path::Path, bvec_path: &std::path::Path) -> Option<GradientScheme> {
    if !bval_path.exists() || !bvec_path.exists() {
        return None;
    }
    let bval = std::fs::read_to_string(bval_path).ok()?;
    let bvec = std::fs::read_to_string(bvec_path).ok()?;
    let scheme = read_fsl_scheme(&bval, &bvec).ok()?;
    normalise_scheme(scheme)
}

// ── Multi-dataset cross-codec tests ──────────────────────────────────────

/// Cross-codec round-trip on ds002087.  Subsets to b≥2000 (single shell)
/// because the NRRD codec uses one nominal `DWMRI_b-value`.
#[test]
#[ignore = "requires downloaded ds002087 dataset"]
fn multi_dataset_cross_codec_ds002087() {
    let dwi = ds002087_dwi_dir();
    let scheme = load_scheme_from_fsl(
        &dwi.join("sub-01_run-1_dwi.bval"),
        &dwi.join("sub-01_run-1_dwi.bvec"),
    )
    .expect("ds002087 scheme must load");

    assert!(scheme.len() >= 50, "ds002087 must have ≥50 volumes");
    // ds002087 is mixed-shell (b≈700 + b≈2000); subset to single shell.
    let single = subset_by_b(&scheme, 2_000.0, 50.0);
    assert!(single.len() >= 30, "single-shell subset must have ≥30 volumes");

    let [fsl, mrtrix, nrrd, dicom] = cross_codec_round_trip_all(&single);

    // Per-codec frame conventions.
    assert_eq!(fsl.frame(), GradientFrame::ImageAxis);
    assert_eq!(mrtrix.frame(), GradientFrame::ImageAxis);
    assert_eq!(nrrd.frame(), GradientFrame::Lps);
    assert_eq!(dicom.frame(), GradientFrame::Lps);

    // All four must agree (transitive: FSL is the reference).
    for other in [&mrtrix, &nrrd, &dicom] {
        assert_scheme_equivalence(&fsl, other);
    }
}

/// Cross-codec round-trip on ds004666 (EDDEN).  Subsets to b≥2000 for
/// NRRD single-shell compatibility.
#[test]
#[ignore = "requires downloaded ds004666 dataset"]
fn multi_dataset_cross_codec_ds004666() {
    let dwi = ds004666_dwi_dir();
    let scheme = load_scheme_from_fsl(
        &dwi.join("sub-01_ses-0p9mm_dir-AP_dwi.bval"),
        &dwi.join("sub-01_ses-0p9mm_dir-AP_dwi.bvec"),
    )
    .expect("ds004666 scheme must load");

    assert!(scheme.len() >= 100, "ds004666 must have ≥100 volumes");
    let single = subset_by_b(&scheme, 2_000.0, 50.0);
    assert!(single.len() >= 30, "single-shell subset must have ≥30 volumes");

    let [fsl, mrtrix, nrrd, dicom] = cross_codec_round_trip_all(&single);

    // Per-codec frame conventions.
    assert_eq!(fsl.frame(), GradientFrame::ImageAxis);
    assert_eq!(mrtrix.frame(), GradientFrame::ImageAxis);
    assert_eq!(nrrd.frame(), GradientFrame::Lps);
    assert_eq!(dicom.frame(), GradientFrame::Lps);

    // All four must agree (transitive: FSL is the reference).
    for other in [&mrtrix, &nrrd, &dicom] {
        assert_scheme_equivalence(&fsl, other);
    }
}
