use std::io::Write;
use std::path::Path;

use anyhow::Result;
use ritk_diffusion_scheme::{DiffusionWeighting, GradientFrame};
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

fn weighting(value: f64) -> DiffusionWeighting {
    DiffusionWeighting::from_seconds_per_square_millimeter(value).expect("finite weighting")
}

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
