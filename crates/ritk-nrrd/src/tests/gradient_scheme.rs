use std::io::Write;
use std::path::Path;

use anyhow::Result;
use ritk_diffusion_scheme::{DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme};
use ritk_spatial::Vector;
use tempfile::tempdir;

use crate::read_nrrd_gradient_scheme;

fn write_header(path: &Path, fields: &[&str]) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "NRRD0005")?;
    writeln!(file, "type: float")?;
    writeln!(file, "dimension: 4")?;
    writeln!(file, "space: left-posterior-superior")?;
    writeln!(file, "sizes: 3 2 2 2")?;
    writeln!(file, "space directions: none (1,0,0) (0,1,0) (0,0,1)")?;
    writeln!(file, "kinds: list domain domain domain")?;
    writeln!(file, "encoding: raw")?;
    for field in fields {
        writeln!(file, "{field}")?;
    }
    writeln!(file)?;
    Ok(())
}

/// Write NRRD header fields for a gradient scheme suitable for
/// [`read_nrrd_gradient_scheme`].  Returns the DWMRI_b-value and
/// DWMRI_gradient_NNNN lines.
///
/// The NRRD DWI convention encodes effective b-values via gradient
/// magnitude: `b_eff = nominal * (|g| / max|g|)²`.  This helper computes
/// the maximum b-value as the nominal and scales each raw direction
/// accordingly.  b0 entries are emitted as the zero vector.
fn nrrd_scheme_fields(scheme: &GradientScheme) -> Vec<String> {
    let max_b = scheme
        .directions()
        .iter()
        .map(|entry| entry.weighting().seconds_per_square_millimeter())
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);

    let mut fields: Vec<String> = Vec::new();
    let count = scheme.len();
    fields.push(format!("sizes: {count} 2 2 2"));
    fields.push("modality:=DWMRI".to_owned());
    fields.push(format!("DWMRI_b-value:={max_b}"));
    for (index, entry) in scheme.directions().iter().enumerate() {
        let b = entry.weighting().seconds_per_square_millimeter();
        let [x, y, z] = entry.direction().to_array();
        if b == 0.0 {
            fields.push(format!("DWMRI_gradient_{index:04}:=0 0 0"));
        } else {
            let scale = (b / max_b).sqrt();
            fields.push(format!(
                "DWMRI_gradient_{index:04}:={} {} {}",
                x * scale,
                y * scale,
                z * scale
            ));
        }
    }
    fields
}

fn weighting(value: f64) -> DiffusionWeighting {
    DiffusionWeighting::from_seconds_per_square_millimeter(value).expect("finite weighting")
}

// ── ADR 0036 verification condition 8: NRRD round-trip ──────────────────

#[test]
fn nrrd_write_read_round_trip_recovers_identical_scheme() -> Result<()> {
    let scheme = GradientScheme::new(
        vec![
            GradientDirection::new(weighting(0.0), Vector::new([0.0, 0.0, 0.0])).unwrap(),
            GradientDirection::new(
                weighting(500.0),
                Vector::new([0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0]),
            )
            .unwrap(),
            GradientDirection::new(weighting(1_000.0), Vector::new([0.0, 1.0, 0.0])).unwrap(),
            GradientDirection::new(weighting(2_000.0), Vector::new([0.0, 0.0, 1.0])).unwrap(),
        ],
        GradientFrame::Lps,
    )?;

    let directory = tempdir()?;
    let path = directory.path().join("roundtrip.nrrd");
    let fields = nrrd_scheme_fields(&scheme);
    let field_strs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
    write_header(&path, &field_strs)?;

    let recovered = read_nrrd_gradient_scheme(path)?;

    assert_eq!(recovered.frame(), scheme.frame());
    assert_eq!(recovered.len(), scheme.len());
    for (original, recovered) in scheme
        .directions()
        .iter()
        .zip(recovered.directions().iter())
    {
        let delta = (original.weighting().seconds_per_square_millimeter()
            - recovered.weighting().seconds_per_square_millimeter())
        .abs();
        assert!(
            delta < 1e-9,
            "weightings differ by {delta}: original {:?}, recovered {:?}",
            original.weighting().seconds_per_square_millimeter(),
            recovered.weighting().seconds_per_square_millimeter(),
        );
        assert!(
            (original.direction().to_array()[0] - recovered.direction().to_array()[0]).abs() < 1e-9,
        );
        assert!(
            (original.direction().to_array()[1] - recovered.direction().to_array()[1]).abs() < 1e-9,
        );
        assert!(
            (original.direction().to_array()[2] - recovered.direction().to_array()[2]).abs() < 1e-9,
        );
    }
    Ok(())
}

// ── Original tests ───────────────────────────────────────────────────────

#[test]
fn nominal_weighting_and_gradient_magnitude_form_multiple_shells() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("multishell.nrrd");
    write_header(
        &path,
        &[
            "modality:=DWMRI",
            "DWMRI_b-value:=1000",
            "DWMRI_gradient_0000:=0 0 0",
            "DWMRI_gradient_0001:=0.7071067811865476 0 0",
            "DWMRI_gradient_0002:=0 1 0",
        ],
    )?;
    let scheme = read_nrrd_gradient_scheme(path)?;
    assert_eq!(scheme.frame(), GradientFrame::Lps);
    assert_eq!(scheme.directions()[0].weighting(), weighting(0.0));
    assert!(
        (scheme.directions()[1]
            .weighting()
            .seconds_per_square_millimeter()
            - 500.0)
            .abs()
            < 1.0e-9
    );
    assert_eq!(scheme.directions()[2].weighting(), weighting(1_000.0));
    assert_eq!(
        scheme.directions()[1].direction(),
        Vector::new([1.0, 0.0, 0.0])
    );
    Ok(())
}

#[test]
fn low_effective_weighting_is_canonicalized_to_b0() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("low_effective_weighting.nrrd");
    write_header(
        &path,
        &[
            "modality:=DWMRI",
            "DWMRI_b-value:=1000",
            "DWMRI_gradient_0000:=0 0 0",
            "DWMRI_gradient_0001:=0.2 0 0",
            "DWMRI_gradient_0002:=1 0 0",
        ],
    )?;

    let scheme = read_nrrd_gradient_scheme(path)?;
    assert_eq!(scheme.directions()[1].weighting(), weighting(0.0));
    assert_eq!(
        scheme.directions()[1].direction(),
        Vector::new([0.0, 0.0, 0.0])
    );
    assert_eq!(scheme.directions()[2].weighting(), weighting(1_000.0));
    Ok(())
}

#[test]
fn measurement_frame_and_ras_space_convert_once_to_lps() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("frame.nrrd");
    write_header(
        &path,
        &[
            "space: right-anterior-superior",
            "measurement frame: (0,1,0) (1,0,0) (0,0,1)",
            "modality:=DWMRI",
            "DWMRI_b-value:=1000",
            "DWMRI_gradient_0000:=0 0 0",
            "DWMRI_gradient_0001:=1 0 0",
            "DWMRI_gradient_0002:=0 1 0",
        ],
    )?;
    let scheme = read_nrrd_gradient_scheme(path)?;
    assert_eq!(
        scheme.directions()[1].direction(),
        Vector::new([0.0, -1.0, 0.0])
    );
    assert_eq!(
        scheme.directions()[2].direction(),
        Vector::new([-1.0, 0.0, 0.0])
    );
    Ok(())
}

#[test]
fn missing_and_malformed_dwi_contracts_are_rejected() -> Result<()> {
    let directory = tempdir()?;
    let missing = directory.path().join("missing.nrrd");
    write_header(&missing, &["modality:=DWMRI", "DWMRI_b-value:=1000"])?;
    assert!(read_nrrd_gradient_scheme(missing).is_err());

    let list = directory.path().join("list.nrrd");
    write_header(
        &list,
        &[
            "modality:=DWMRI",
            "DWMRI_b-value:=0 1000 1000",
            "DWMRI_gradient_0000:=0 0 0",
            "DWMRI_gradient_0001:=1 0 0",
            "DWMRI_gradient_0002:=0 1 0",
        ],
    )?;
    assert!(read_nrrd_gradient_scheme(list).is_err());

    let non_finite = directory.path().join("non_finite.nrrd");
    write_header(
        &non_finite,
        &[
            "modality:=DWMRI",
            "DWMRI_b-value:=1000",
            "DWMRI_gradient_0000:=0 0 0",
            "DWMRI_gradient_0001:=NaN 0 0",
            "DWMRI_gradient_0002:=0 1 0",
        ],
    )?;
    assert!(read_nrrd_gradient_scheme(non_finite).is_err());
    Ok(())
}

#[test]
fn nex_and_b_matrix_encodings_fail_explicitly() -> Result<()> {
    let directory = tempdir()?;
    for (name, extra) in [
        ("nex", "DWMRI_NEX_0000:=2"),
        ("matrix", "DWMRI_B-matrix_0000:=1 0 0 1 0 1"),
    ] {
        let path = directory.path().join(format!("{name}.nrrd"));
        write_header(
            &path,
            &[
                "modality:=DWMRI",
                "DWMRI_b-value:=1000",
                "DWMRI_gradient_0000:=0 0 0",
                "DWMRI_gradient_0001:=1 0 0",
                "DWMRI_gradient_0002:=0 1 0",
                extra,
            ],
        )?;
        assert!(read_nrrd_gradient_scheme(path).is_err());
    }
    Ok(())
}

#[test]
fn gradient_count_must_match_acquisition_extent() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("count_mismatch.nrrd");
    write_header(
        &path,
        &[
            "sizes: 4 2 2 2",
            "modality:=DWMRI",
            "DWMRI_b-value:=1000",
            "DWMRI_gradient_0000:=0 0 0",
            "DWMRI_gradient_0001:=1 0 0",
            "DWMRI_gradient_0002:=0 1 0",
        ],
    )?;
    let error = read_nrrd_gradient_scheme(path).expect_err("gradient count mismatch");
    assert!(error
        .to_string()
        .contains("does not match acquisition-axis extent 4"));
    Ok(())
}
