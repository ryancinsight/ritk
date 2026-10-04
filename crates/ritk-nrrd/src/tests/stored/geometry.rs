//! Stored NRRD geometry tests.

use super::*;

#[test]
fn stored_reader_normalizes_patient_spaces_and_physical_units() -> Result<()> {
    let directory = tempdir()?;
    let ras_path = directory.path().join("ras-centimeters.nrrd");
    let ras_fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "space: RAS",
        "space units: \"cm\" \"cm\" \"cm\"",
        "space directions: (1,0,0) (0,2,0) (0,0,3)",
        "space origin: (1,2,3)",
    ];
    write_header(&ras_path, &ras_fields, &[7])?;
    let ras = read_nrrd_stored(&ras_path)?;
    assert_eq!(ras.metadata().origin().to_array(), [-10.0, -20.0, 30.0]);
    assert_eq!(ras.metadata().spacing().to_array(), [30.0, 20.0, 10.0]);
    assert_eq!(
        *ras.metadata().direction(),
        Direction::from_rows([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    );

    let las_path = directory.path().join("las-millimeters.nrrd");
    let las_fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "space: LAS",
        "space units: \"mm\" \"mm\" \"mm\"",
        "space directions: (1,0,0) (0,2,0) (0,0,3)",
        "space origin: (1,2,3)",
    ];
    write_header(&las_path, &las_fields, &[9])?;
    let las = read_nrrd_stored(&las_path)?;
    assert_eq!(las.metadata().origin().to_array(), [1.0, -2.0, 3.0]);
    assert_eq!(las.metadata().spacing().to_array(), [3.0, 2.0, 1.0]);
    assert_eq!(
        *las.metadata().direction(),
        Direction::from_rows([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    );
    Ok(())
}

#[test]
fn stored_reader_preserves_orientation_for_small_positive_spacings() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("small-spacings.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "space: LPS",
        "space units: \"mm\" \"mm\" \"mm\"",
        "space directions: (-1e-10,0,0) (0,-1e-10,0) (0,0,1e-10)",
    ];
    write_header(&path, &fields, &[5])?;

    let volume = read_nrrd_stored(&path)?;
    for spacing in volume.metadata().spacing().to_array() {
        assert!((spacing - 1e-10).abs() <= f64::EPSILON * 1e-10);
    }
    assert_eq!(
        volume.metadata().direction().to_row_major(),
        [0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0]
    );
    assert_eq!(
        volume
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [5]
    );
    Ok(())
}

#[test]
fn stored_reader_rejects_finite_competing_spacing_and_invalid_spacing() -> Result<()> {
    let directory = tempdir()?;
    for (spacings, expected_conflict) in [("0.5 0.5 0.5", true), ("not a spacing list", false)] {
        let path = directory.path().join("conflicting-spatial-fields.nrrd");
        let fields = [
            "type: unsigned char",
            "dimension: 3",
            "sizes: 1 1 1",
            "space directions: (1,0,0) (0,1,0) (0,0,1)",
            &format!("spacings: {spacings}"),
        ];
        write_header(&path, &fields, &[7])?;

        let error = read_nrrd_stored(&path).expect_err("invalid competing spacing metadata");
        if expected_conflict {
            assert!(matches!(
                error,
                NrrdStoredReadError::ConflictingSpatialFields
            ));
        } else {
            assert!(matches!(
                error,
                NrrdStoredReadError::SpatialMetadata {
                    field: NrrdSpatialMetadataField::Spacings,
                    ..
                }
            ));
        }
    }
    Ok(())
}

#[test]
fn stored_reader_rejects_unrepresented_axis_bounds_before_payload_read() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("axis-bounds.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 2 2 2",
        "spacings: 1 1 1",
        "axis mins: 0 0 0",
        "axis maxs: 1 1 1",
    ];
    write_header(&path, &fields, &[])?;

    let error = read_nrrd_stored(&path).expect_err("axis support bounds are not in StoredVolume");
    assert!(matches!(
        error,
        NrrdStoredReadError::SpatialMetadata {
            field: NrrdSpatialMetadataField::AxisBounds,
            ..
        }
    ));
    Ok(())
}

#[test]
fn stored_reader_rejects_unrepresented_centering_before_payload_read() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("cell-centered.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "space directions: (1,0,0) (0,1,0) (0,0,1)",
        "centers: \"cell\" \"cell\" \"cell\"",
    ];
    write_header(&path, &fields, &[])?;

    let error = read_nrrd_stored(&path).expect_err("cell support is not in StoredVolume");
    assert!(matches!(
        error,
        NrrdStoredReadError::SpatialMetadata {
            field: NrrdSpatialMetadataField::Centering,
            ..
        }
    ));
    Ok(())
}

#[test]
fn stored_reader_rejects_sample_units_before_payload_read() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("sample-units.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "sample units: \"HU\"",
    ];
    write_header(&path, &fields, &[])?;

    assert!(matches!(
        read_nrrd_stored(&path),
        Err(NrrdStoredReadError::UnsupportedSampleUnits { .. })
    ));
    Ok(())
}

#[test]
fn stored_reader_accepts_nan_spacings_along_direction_axes() -> Result<()> {
    let directory = tempdir()?;
    let path = directory
        .path()
        .join("directions-with-unspecified-spacings.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "space directions: (2,0,0) (0,3,0) (0,0,4)",
        "spacings: nan nan nan",
    ];
    write_header(&path, &fields, &[17])?;

    let volume = read_nrrd_stored(&path)?;
    assert_eq!(volume.metadata().spacing().to_array(), [4.0, 3.0, 2.0]);
    assert_eq!(
        volume
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [17]
    );
    Ok(())
}

#[test]
fn stored_reader_converts_per_axis_spacing_units_to_millimeters() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("per-axis-spacing-units.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "space: LPS",
        "space units: \"mm\" \"mm\" \"mm\"",
        "spacings: 1 2 3",
        "units: \"cm\" \"mm\" \"m\"",
    ];
    write_header(&path, &fields, &[23])?;

    let volume = read_nrrd_stored(&path)?;
    assert_eq!(volume.metadata().spacing().to_array(), [3_000.0, 2.0, 10.0]);
    // NRRD axes are i,j,k; StoredVolume axes are depth,row,column = k,j,i.
    assert_eq!(
        *volume.metadata().direction(),
        Direction::from_rows([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    );
    Ok(())
}

#[test]
fn stored_reader_rejects_per_axis_units_without_grid_spacing() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("units-without-spacing.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "units: \"cm\" \"mm\" \"m\"",
    ];
    write_header(&path, &fields, &[])?;

    let error = read_nrrd_stored(&path)
        .expect_err("axis units need a grid spacing before physical geometry is representable");
    assert!(matches!(
        error,
        NrrdStoredReadError::SpatialMetadata {
            field: NrrdSpatialMetadataField::AxisUnits,
            ..
        }
    ));
    Ok(())
}

#[test]
fn stored_reader_rejects_singular_coordinate_map_before_payload_read() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("singular-coordinate-map.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 2 2 1",
        "ritk_coordinate_map:=slice_series count=1 transforms=0,0,0,0,0,0,0,0,0,0,0,0",
    ];
    write_header(&path, &fields, &[])?;

    let error = read_nrrd_stored(&path).expect_err("singular map must reject before payload use");
    assert!(matches!(error, NrrdStoredReadError::CoordinateMap { .. }));
    assert_eq!(
        error.to_string(),
        "NRRD coordinate map is invalid: slice 0 rotation matrix is singular"
    );
    Ok(())
}

#[test]
fn stored_reader_rejects_unknown_geometry_endian_and_space_dimension() -> Result<()> {
    let directory = tempdir()?;
    let invalid_endian = directory.path().join("invalid-endian.nrrd");
    let invalid_endian_fields = [
        "type: unsigned short",
        "dimension: 3",
        "sizes: 1 1 1",
        "endian: bi g",
        "encoding: raw",
    ];
    write_header(&invalid_endian, &invalid_endian_fields, &[1, 2])?;
    let endian_error = read_nrrd_stored(&invalid_endian)
        .expect_err("an unknown explicit endian marker must not decode as little-endian");
    assert!(matches!(
        endian_error,
        NrrdStoredReadError::InvalidByteOrder { .. }
    ));

    let missing_multibyte_endian = directory.path().join("missing-endian.nrrd");
    let missing_endian_fields = [
        "type: unsigned short",
        "dimension: 3",
        "sizes: 1 1 1",
        "encoding: raw",
    ];
    write_header(&missing_multibyte_endian, &missing_endian_fields, &[1, 2])?;
    let missing_endian_error = read_nrrd_stored(&missing_multibyte_endian)
        .expect_err("multi-byte binary samples require a declared byte order");
    assert!(matches!(
        missing_endian_error,
        NrrdStoredReadError::MissingByteOrder { sample_width: 2 }
    ));

    let single_byte_without_endian = directory.path().join("single-byte-no-endian.nrrd");
    let single_byte_fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "encoding: raw",
    ];
    write_header(&single_byte_without_endian, &single_byte_fields, &[7])?;
    let single_byte = read_nrrd_stored(&single_byte_without_endian)?;
    assert_eq!(single_byte.samples().sample_type(), SampleType::U8);
    assert_eq!(
        single_byte
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [7]
    );

    let invalid_space = directory.path().join("anonymous-space.nrrd");
    let invalid_space_fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "space dimension: 3",
        "space origin: (0,0,0)",
    ];
    write_header(&invalid_space, &invalid_space_fields, &[4])?;
    let space_error = read_nrrd_stored(&invalid_space)
        .expect_err("an anonymous coordinate basis cannot be labeled patient LPS");
    assert!(matches!(
        space_error,
        NrrdStoredReadError::SpatialMetadata {
            field: NrrdSpatialMetadataField::CoordinateSystem,
            ..
        }
    ));

    let invalid_units = directory.path().join("unknown-space-unit.nrrd");
    let invalid_units_fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "space: LPS",
        "space units: \"parsec\" \"mm\" \"mm\"",
        "space directions: (1,0,0) (0,1,0) (0,0,1)",
    ];
    write_header(&invalid_units, &invalid_units_fields, &[5])?;
    let units_error = read_nrrd_stored(&invalid_units)
        .expect_err("unsupported units must not be silently relabeled as millimeters");
    assert!(matches!(
        units_error,
        NrrdStoredReadError::SpatialMetadata {
            field: NrrdSpatialMetadataField::CoordinateSystem,
            ..
        }
    ));

    let anonymous_rank_two = directory.path().join("anonymous-rank-two.nrrd");
    let anonymous_rank_two_fields = [
        "type: unsigned char",
        "dimension: 2",
        "sizes: 1 1",
        "space dimension: 3",
        "space directions: (1,0,0) (0,1,0)",
        "space origin: (0,0,0)",
    ];
    write_header(&anonymous_rank_two, &anonymous_rank_two_fields, &[6])?;
    let anonymous_error = read_nrrd_stored(&anonymous_rank_two)
        .expect_err("array rank does not determine the declared world-space basis");
    assert!(matches!(
        anonymous_error,
        NrrdStoredReadError::SpatialMetadata {
            field: NrrdSpatialMetadataField::CoordinateSystem,
            ..
        }
    ));
    Ok(())
}

#[test]
fn stored_rank_two_named_space_uses_three_dimensional_world_coordinates() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("rank-two-lps.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 2",
        "sizes: 1 1",
        "space: LPS",
        "space directions: (0.5,0,0) (0,1.2,1.6)",
        "space origin: (3,4,5)",
    ];
    write_header(&path, &fields, &[7])?;
    let volume = read_nrrd_stored(&path)?;
    assert_eq!(volume.shape(), [1, 1, 1]);
    assert_eq!(volume.metadata().origin().to_array(), [3.0, 4.0, 5.0]);
    assert_eq!(volume.metadata().spacing().to_array(), [1.0, 2.0, 0.5]);
    assert_eq!(
        volume.metadata().direction().to_row_major(),
        [0.0, 0.0, 1.0, -0.8, 0.6, 0.0, 0.6, 0.8, 0.0]
    );

    let malformed = directory.path().join("rank-two-short-world-vectors.nrrd");
    let malformed_fields = [
        "type: unsigned char",
        "dimension: 2",
        "sizes: 1 1",
        "space: LPS",
        "space directions: (1,0) (0,1)",
        "space origin: (0,0)",
    ];
    write_header(&malformed, &malformed_fields, &[9])?;
    let error = read_nrrd_stored(&malformed)
        .expect_err("named three-dimensional space needs three-component vectors");
    assert!(matches!(
        error,
        NrrdStoredReadError::SpatialMetadata {
            field: NrrdSpatialMetadataField::SpaceDirections,
            ..
        }
    ));
    Ok(())
}
