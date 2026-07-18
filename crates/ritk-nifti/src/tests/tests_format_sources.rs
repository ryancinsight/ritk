//! NIfTI format-source and import coverage.

use super::*;
use std::path::PathBuf;

struct RepositoryFixtureSource {
    name: &'static str,
    relative_path: &'static str,
    format: &'static str,
    source: &'static str,
    license: &'static str }

const REPOSITORY_NIFTI_SOURCES: &[RepositoryFixtureSource] = &[RepositoryFixtureSource {
    name: "MNI152 atlas copy used as brain_fixed",
    relative_path: "../../test_data/registration/brain_fixed.nii.gz",
    format: "NIfTI-1 gzip single-file",
    source: "ANTs example MNI152 atlas, copied from test_data/ants_example/mni152.nii.gz; documented in test_data/registration/README.md",
    license: "Per ANTs/MNI152 distribution terms; see test_data/README.md licensing table" }];

fn repository_path(relative_path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative_path)
}

#[test]
fn repository_nifti_fixture_sources_are_documented() {
    for source in REPOSITORY_NIFTI_SOURCES {
        let path = repository_path(source.relative_path);
        assert!(
            path.exists(),
            "{} fixture path must exist: {}",
            source.name,
            path.display()
        );
        assert!(
            source.format.contains("NIfTI"),
            "{} must name the NIfTI format variant",
            source.name
        );
        assert!(
            source.source.contains("test_data"),
            "{} source must link back to the repository data manifest",
            source.name
        );
        assert!(
            !source.license.trim().is_empty(),
            "{} must document license/provenance terms",
            source.name
        );
    }
}

#[test]
fn imports_sourced_repository_nifti1_gzip_fixture() -> Result<()> {
    let backend = SequentialBackend;
    let source = &REPOSITORY_NIFTI_SOURCES[0];
    let path = repository_path(source.relative_path);

    let loaded = crate::read_nifti(&path, &backend)?;

    assert_eq!(
        loaded.shape(),
        [215, 256, 207],
        "MNI152 NIfTI-1 gzip fixture must import as RITK ZYX shape"
    );
    for axis in 0..3 {
        assert!(
            (loaded.spacing()[axis] - 0.737_463_116_645_813).abs() < 1.0e-6,
            "MNI152 spacing axis {axis} must match sourced fixture metadata"
        );
    }

    let data = loaded.data_slice().expect("contiguous host data").to_vec();
    assert_eq!(
        data.len(),
        215 * 256 * 207,
        "imported MNI152 voxel count must equal sourced header dimensions"
    );
    assert!(
        data.iter().any(|value| *value > 0.0),
        "sourced MNI152 fixture must import non-zero image content"
    );
    assert!(
        data.iter().all(|value| value.is_finite()),
        "sourced MNI152 fixture must import finite Float32 voxels"
    );

    Ok(())
}

#[test]
fn imports_generated_nifti2_gzip_fixture() -> Result<()> {
    let dir = tempdir()?;
    let file_path = dir.path().join("synthetic_nifti2.nii.gz");
    let backend = SequentialBackend;

    let values = (0..24)
        .map(|index| index as f32 * 0.5 + 1.0)
        .collect::<Vec<_>>();
    let image = Image::from_flat_on(
        values.clone(),
        [2, 3, 4],
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.8, 1.2, 1.6]),
        Direction::identity(),
        &backend,
    )
    .expect("valid image dimensions");

    crate::write_nifti2(&file_path, &image, &backend)?;
    let bytes = std::fs::read(&file_path)?;
    assert_eq!(
        &bytes[..2],
        &[0x1f, 0x8b],
        "generated NIfTI-2 gzip fixture must be gzip-wrapped"
    );

    let loaded = crate::read_nifti(&file_path, &backend)?;
    assert_eq!(loaded.shape(), [2, 3, 4]);
    assert_eq!(
        loaded.data_slice().expect("contiguous host data").to_vec(),
        values,
        "generated NIfTI-2 gzip fixture must preserve voxel values"
    );

    Ok(())
}

#[test]
fn imports_generated_uint8_nifti1_fixture() -> Result<()> {
    let backend = SequentialBackend;
    let header = NiftiHeader::new_3d(
        HeaderDims {
            nx: 4,
            ny: 3,
            nz: 2 },
        NiftiDatatype::Uint8,
        HeaderSpatial {
            pixdim: [1.0, 0.5, 0.75, 1.25, 1.0, 1.0, 1.0, 1.0],
            srow_x: [-0.5, 0.0, 0.0, 1.0],
            srow_y: [0.0, -0.75, 0.0, 2.0],
            srow_z: [0.0, 0.0, 1.25, 3.0] },
    )?;
    let values = (0..24).map(|value| value as u8).collect::<Vec<_>>();
    let bytes = write_single_file_bytes(&header, &values);

    let loaded = crate::read_nifti_from_bytes(&bytes, &backend)?;

    assert_eq!(loaded.shape(), [2, 3, 4]);
    assert_eq!(
        loaded.data_slice().expect("contiguous host data").to_vec(),
        values.iter().copied().map(f32::from).collect::<Vec<_>>(),
        "UInt8 NIfTI-1 image voxels must import into f32 tensor values without reordering"
    );

    Ok(())
}

#[test]
fn analyze_style_header_is_not_imported_as_nifti() {
    let backend = SequentialBackend;
    let mut analyze_header = vec![0_u8; 348];
    analyze_header[0..4].copy_from_slice(&348_i32.to_le_bytes());

    let err = crate::read_nifti_from_bytes(&analyze_header, &backend)
        .expect_err("Analyze 7.5 header without NIfTI magic must be rejected");
    let msg = format!("{err:#}");

    assert!(
        msg.contains("Unsupported NIfTI-1 magic"),
        "NIfTI reader must reject Analyze-style .hdr bytes instead of importing them: {msg}"
    );
}
