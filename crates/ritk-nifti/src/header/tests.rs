use super::{HeaderAxis, HeaderDims, HeaderSpatial, HeaderVersion, NiftiDatatype, NiftiHeader};

#[test]
fn header_round_trip_preserves_nifti1_core_fields() {
    let header = NiftiHeader::new_volume(
        HeaderDims {
            nx: 4,
            ny: 3,
            nz: 2,
        },
        NiftiDatatype::Float32,
        HeaderSpatial {
            pixdim: [1.0, 0.75, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0],
            srow_x: [-0.75, 0.0, 0.0, -11.0],
            srow_y: [0.0, -1.5, 0.0, 7.5],
            srow_z: [0.0, 0.0, 2.0, 3.25],
        },
    )
    .expect("valid header");

    let parsed = NiftiHeader::parse(&header.encode()).expect("encoded header parses");
    assert_eq!(parsed.version, HeaderVersion::One);
    assert_eq!(parsed.dim, [3, 4, 3, 2, 1, 1, 1, 1]);
    assert_eq!(parsed.datatype, NiftiDatatype::Float32);
    assert_eq!(parsed.srow_x, [-0.75, 0.0, 0.0, -11.0]);
    assert_eq!(parsed.vox_offset, 352);
}

#[test]
fn header_round_trip_preserves_nifti2_core_fields() {
    let header = NiftiHeader::new_with_version(
        HeaderVersion::Two,
        HeaderDims {
            nx: 70_000,
            ny: 3,
            nz: 2,
        },
        1,
        HeaderAxis::Volume,
        NiftiDatatype::Uint32,
        HeaderSpatial {
            pixdim: [1.0, 0.75, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0],
            srow_x: [-0.75, 0.0, 0.0, -11.0],
            srow_y: [0.0, -1.5, 0.0, 7.5],
            srow_z: [0.0, 0.0, 2.0, 3.25],
        },
    )
    .expect("valid header");

    let parsed = NiftiHeader::parse(&header.encode()).expect("encoded header parses");
    assert_eq!(parsed.version, HeaderVersion::Two);
    assert_eq!(parsed.dim, [3, 70_000, 3, 2, 1, 1, 1, 1]);
    assert_eq!(parsed.datatype, NiftiDatatype::Uint32);
    assert_eq!(parsed.srow_x, [-0.75, 0.0, 0.0, -11.0]);
    assert_eq!(parsed.vox_offset, 544);
}

#[test]
fn acquisition_axis_remains_rank_four_with_one_value() {
    let spatial = HeaderSpatial {
        pixdim: [1.0; 8],
        srow_x: [1.0, 0.0, 0.0, 0.0],
        srow_y: [0.0, 1.0, 0.0, 0.0],
        srow_z: [0.0, 0.0, 1.0, 0.0],
    };

    for version in [HeaderVersion::One, HeaderVersion::Two] {
        let header = NiftiHeader::new_with_version(
            version,
            HeaderDims {
                nx: 1,
                ny: 1,
                nz: 1,
            },
            1,
            HeaderAxis::Acquisition,
            NiftiDatatype::Uint8,
            spatial,
        )
        .expect("one-entry acquisition axis is valid");

        assert_eq!(header.dim[0], 4, "acquisition axes remain rank four");
        assert_eq!(header.dim[4], 1, "singleton acquisition count is retained");
    }
}

#[test]
fn header_round_trip_preserves_versioned_precision_and_scaling() {
    let srow_x = [0.123_456_789_012_345, 0.0, 0.0, 123_456.789_012_345];
    let spatial = HeaderSpatial {
        pixdim: [1.0, 0.123_456_789_012_345, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0],
        srow_x,
        srow_y: [0.0, -1.5, 0.0, 7.5],
        srow_z: [0.0, 0.0, 2.0, 3.25],
    };

    for version in [HeaderVersion::One, HeaderVersion::Two] {
        let mut header = NiftiHeader::new_with_version(
            version,
            HeaderDims {
                nx: 1,
                ny: 1,
                nz: 1,
            },
            1,
            HeaderAxis::Volume,
            NiftiDatatype::Uint8,
            spatial,
        )
        .expect("one-voxel header is valid");
        header.scl_slope = 2.5;
        header.scl_inter = -17.25;

        let parsed = NiftiHeader::parse(&header.encode()).expect("encoded header parses");

        assert_eq!(parsed.scl_slope, 2.5);
        assert_eq!(parsed.scl_inter, -17.25);
        match version {
            HeaderVersion::One => {
                assert_eq!(
                    parsed.srow_x[0],
                    f64::from(
                        super::convert::checked_f64_to_f32(srow_x[0], "srow")
                            .expect("test value fits NIfTI-1")
                    )
                );
            }
            HeaderVersion::Two => assert_eq!(parsed.srow_x, srow_x),
        }
    }
}

#[test]
fn nifti1_rejects_dimensions_above_signed_i16() {
    let err = NiftiHeader::new_volume(
        HeaderDims {
            nx: 70_000,
            ny: 1,
            nz: 1,
        },
        NiftiDatatype::Float32,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
        },
    )
    .expect_err("NIfTI-1 dimensions above signed i16 must be rejected");

    assert!(
        err.to_string().contains("i16"),
        "error must name NIfTI-1 signed dimension bound: {err}"
    );
}

#[test]
fn nifti1_dimension_boundary_matches_signed_i16() {
    let spatial = HeaderSpatial {
        pixdim: [1.0; 8],
        srow_x: [1.0, 0.0, 0.0, 0.0],
        srow_y: [0.0, 1.0, 0.0, 0.0],
        srow_z: [0.0, 0.0, 1.0, 0.0],
    };
    let maximum = NiftiHeader::new_volume(
        HeaderDims {
            nx: 32_767,
            ny: 1,
            nz: 1,
        },
        NiftiDatatype::Uint8,
        spatial,
    )
    .expect("the positive signed-i16 boundary is representable");
    assert_eq!(maximum.dim[1], 32_767);

    let error = NiftiHeader::new_volume(
        HeaderDims {
            nx: 32_768,
            ny: 1,
            nz: 1,
        },
        NiftiDatatype::Uint8,
        spatial,
    )
    .expect_err("one above signed-i16 maximum is not representable");
    assert!(error.to_string().contains("i16"));
}

#[test]
fn nifti1_rejects_negative_header_dimensions() {
    let header = NiftiHeader::new_volume(
        HeaderDims {
            nx: 1,
            ny: 1,
            nz: 1,
        },
        NiftiDatatype::Float32,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
        },
    )
    .expect("valid one-voxel header");
    let mut bytes = header.encode();
    bytes[42..44].copy_from_slice(&(-1_i16).to_le_bytes());

    let error = NiftiHeader::parse(&bytes).expect_err("negative NIfTI-1 dimension is invalid");

    assert!(error
        .to_string()
        .contains("dim[1] must be non-negative, got -1"));
}

#[test]
fn newly_mapped_sample_types_reject_image_and_label_conversion() {
    let spatial = HeaderSpatial {
        pixdim: [1.0; 8],
        srow_x: [1.0, 0.0, 0.0, 0.0],
        srow_y: [0.0, 1.0, 0.0, 0.0],
        srow_z: [0.0, 0.0, 1.0, 0.0],
    };
    let datatypes = [
        NiftiDatatype::Int8,
        NiftiDatatype::Uint16,
        NiftiDatatype::Uint64,
        NiftiDatatype::Int64,
        NiftiDatatype::Float64,
    ];

    for version in [HeaderVersion::One, HeaderVersion::Two] {
        for datatype in datatypes {
            let header = NiftiHeader::new_with_version(
                version,
                HeaderDims {
                    nx: 1,
                    ny: 1,
                    nz: 1,
                },
                1,
                HeaderAxis::Volume,
                datatype,
                spatial,
            )
            .expect("one-voxel header is valid");
            let parsed = NiftiHeader::parse(&header.encode()).expect("encoded header parses");
            let sample = vec![0x5a; datatype.byte_width()];

            let image_error = parsed
                .read_f32_voxel(&sample)
                .expect_err("unsupported stored type must not be converted to f32")
                .to_string();
            assert_eq!(
                image_error,
                format!(
                    "NIfTI image convenience reader does not support stored datatype {datatype:?}"
                )
            );

            let label_error = parsed
                .read_label_voxel(&sample)
                .expect_err("unsupported stored type must not be converted to u32")
                .to_string();
            assert_eq!(
                label_error,
                format!(
                    "NIfTI label convenience reader does not support stored datatype {datatype:?}"
                )
            );
        }
    }
}
