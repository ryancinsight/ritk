//! Tests for the `dwi` command group.
//!
//! These cover the command's own responsibilities — argument handling, geometry
//! propagation, and writing the map each flag names. The estimator's value
//! semantics are the library's contract and are tested in
//! `ritk_diffusion::maps`; re-asserting them here would duplicate coverage of
//! someone else's contract.

use super::*;
use ritk_image::tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

/// Isotropic diffusivity of the phantom, in mm²/s.
///
/// An isotropic tensor attenuates as `S = S₀ exp(-b·D)` along every direction,
/// so the expected signal needs no forward model and the expected MD is exactly
/// `D` with FA zero.
const PHANTOM_DIFFUSIVITY: f64 = 8.0e-4;

const SHAPE: [usize; 3] = [2, 2, 2];
const SPACING: [f64; 3] = [2.0, 1.5, 1.5];
const B_VALUE: f64 = 1000.0;

/// Six directions plus one reference, enough to identify six tensor unknowns.
///
/// The three axes and the three diagonals between them. Written through
/// `FRAC_1_SQRT_2` rather than a decimal literal so the diagonals are exactly
/// unit length, which is what the scheme contract requires of them.
const DIAGONAL: f64 = std::f64::consts::FRAC_1_SQRT_2;
const DIRECTIONS: [[f64; 3]; 6] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [DIAGONAL, DIAGONAL, 0.0],
    [DIAGONAL, 0.0, DIAGONAL],
    [0.0, DIAGONAL, DIAGONAL],
];

/// Write an isotropic phantom series and its FSL sidecars.
///
/// Every voxel carries the same tensor, so spatial variation in the output
/// would be a defect in the command's indexing rather than in the data.
fn phantom(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let backend = Backend::default();
    let voxels: usize = SHAPE.iter().product();
    let s0 = 1000.0_f32;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "phantom signals are small and well within f32"
    )]
    let attenuated = (f64::from(s0) * (-B_VALUE * PHANTOM_DIFFUSIVITY).exp()) as f32;

    let volume = |value: f32| {
        Image::new(
            Tensor::<f32, Backend>::from_slice_on(SHAPE, &vec![value; voxels], &backend),
            Point::new([0.0; 3]),
            Spacing::new(SPACING),
            Direction::identity(),
        )
        .expect("invariant: tensor matches the declared rank")
    };

    let mut volumes = vec![volume(s0)];
    volumes.extend(DIRECTIONS.iter().map(|_| volume(attenuated)));

    let dwi = dir.join("dwi.nii");
    ritk_nifti::write_nifti_series(&dwi, &volumes, &backend).expect("write series");

    let bval = dir.join("dwi.bval");
    let mut values = vec!["0".to_owned()];
    values.extend(DIRECTIONS.iter().map(|_| B_VALUE.to_string()));
    std::fs::write(&bval, values.join(" ")).expect("write bval");

    let bvec = dir.join("dwi.bvec");
    let rows: Vec<String> = (0..3)
        .map(|axis| {
            let mut row = vec!["0".to_owned()];
            row.extend(DIRECTIONS.iter().map(|entry| entry[axis].to_string()));
            row.join(" ")
        })
        .collect();
    std::fs::write(&bvec, rows.join("\n")).expect("write bvec");

    (dwi, bval, bvec)
}

fn args(dir: &Path, paths: &(PathBuf, PathBuf, PathBuf)) -> TensorArgs {
    TensorArgs {
        dwi: paths.0.clone(),
        bval: paths.1.clone(),
        bvec: paths.2.clone(),
        fa: Some(dir.join("fa.nii")),
        md: Some(dir.join("md.nii")),
        ad: Some(dir.join("ad.nii")),
        rd: Some(dir.join("rd.nii")),
        // The phantom is uniform, so a percentile-relative mask would threshold
        // it against itself; every voxel here is signal.
        background_fraction: 0.0,
    }
}

fn read(path: &Path) -> Image<f32, Backend, 3> {
    super::super::read_image(path).expect("written map is readable")
}

fn values(image: &Image<f32, Backend, 3>) -> Vec<f32> {
    image.data_slice().expect("contiguous host voxels").to_vec()
}

#[test]
fn isotropic_phantom_yields_its_diffusivity_and_no_anisotropy() {
    let dir = tempdir().expect("tempdir");
    let paths = phantom(dir.path());
    let arguments = args(dir.path(), &paths);
    let fa_path = arguments.fa.clone().expect("fa requested");
    let md_path = arguments.md.clone().expect("md requested");

    tensor(arguments).expect("fit succeeds");

    for value in values(&read(&md_path)) {
        assert!(
            (f64::from(value) - PHANTOM_DIFFUSIVITY).abs() < 1.0e-6,
            "MD {value} should be the phantom's diffusivity {PHANTOM_DIFFUSIVITY}"
        );
    }
    for value in values(&read(&fa_path)) {
        assert!(
            value < 1.0e-3,
            "isotropic diffusion must give FA zero, got {value}"
        );
    }
}

#[test]
fn the_three_diffusivity_maps_describe_one_decomposition() {
    // MD = (AD + 2·RD)/3 by definition, and it holds only if all three maps come
    // from the same eigenvalues. That catches a flag wired to the wrong measure,
    // which a file-existence check cannot.
    let dir = tempdir().expect("tempdir");
    let paths = phantom(dir.path());
    let arguments = args(dir.path(), &paths);
    let md_path = arguments.md.clone().expect("md requested");
    let ad_path = arguments.ad.clone().expect("ad requested");
    let rd_path = arguments.rd.clone().expect("rd requested");

    tensor(arguments).expect("fit succeeds");

    let md = values(&read(&md_path));
    let ad = values(&read(&ad_path));
    let rd = values(&read(&rd_path));
    for ((md, ad), rd) in md.iter().zip(&ad).zip(&rd) {
        let expected = (f64::from(*ad) + 2.0 * f64::from(*rd)) / 3.0;
        assert!(
            (f64::from(*md) - expected).abs() < 1.0e-9,
            "MD {md} should equal (AD {ad} + 2·RD {rd})/3 = {expected}"
        );
        assert!(ad >= rd, "axial {ad} cannot fall below radial {rd}");
    }
}

#[test]
fn written_maps_inherit_the_input_geometry() {
    // A map measures the same voxels. Written into a default frame it would not
    // overlay the anatomy it came from, and no value in it would reveal that.
    let dir = tempdir().expect("tempdir");
    let paths = phantom(dir.path());
    let arguments = args(dir.path(), &paths);
    let fa_path = arguments.fa.clone().expect("fa requested");

    tensor(arguments).expect("fit succeeds");

    let map = read(&fa_path);
    assert_eq!(map.shape(), SHAPE);
    for (axis, expected) in SPACING.iter().enumerate() {
        assert!(
            (map.spacing()[axis] - expected).abs() < 1.0e-9,
            "spacing on axis {axis} should be {expected}, got {}",
            map.spacing()[axis]
        );
    }
}

#[test]
fn requesting_no_output_is_rejected_before_fitting() {
    // Fitting a volume and writing nothing is minutes of work followed by no
    // result, so it fails immediately instead.
    let dir = tempdir().expect("tempdir");
    let paths = phantom(dir.path());
    let mut arguments = args(dir.path(), &paths);
    arguments.fa = None;
    arguments.md = None;
    arguments.ad = None;
    arguments.rd = None;

    let error = tensor(arguments).expect_err("no output requested");
    assert!(
        error.to_string().contains("no output requested"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_scheme_disagreeing_with_the_series_is_rejected() {
    // Both sidecars lose their last entry, so the scheme itself stays valid and
    // simply describes one volume fewer than the series holds. Dropping only
    // the b-value would make the sidecars disagree with each other, which the
    // scheme parser rejects earlier and for a different reason.
    let dir = tempdir().expect("tempdir");
    let paths = phantom(dir.path());

    let bvals: Vec<String> = std::fs::read_to_string(&paths.1)
        .expect("bval")
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    std::fs::write(&paths.1, bvals[..bvals.len() - 1].join(" ")).expect("rewrite bval");

    let rows: Vec<String> = std::fs::read_to_string(&paths.2)
        .expect("bvec")
        .lines()
        .map(|row| {
            let entries: Vec<&str> = row.split_whitespace().collect();
            entries[..entries.len() - 1].join(" ")
        })
        .collect();
    std::fs::write(&paths.2, rows.join("\n")).expect("rewrite bvec");

    let error = tensor(args(dir.path(), &paths)).expect_err("count mismatch");
    assert!(
        error
            .to_string()
            .contains("volumes but the scheme declares"),
        "unexpected error: {error}"
    );
}

#[test]
fn masking_the_whole_volume_is_reported_rather_than_written() {
    // A fraction above one puts the floor above every voxel. Writing four
    // all-zero maps instead would look like a successful run.
    let dir = tempdir().expect("tempdir");
    let paths = phantom(dir.path());
    let mut arguments = args(dir.path(), &paths);
    arguments.background_fraction = 2.0;

    let error = tensor(arguments).expect_err("everything masked");
    assert!(
        error
            .to_string()
            .contains("no voxel yielded an admissible tensor"),
        "unexpected error: {error}"
    );
}
