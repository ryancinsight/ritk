use super::{HeaderDims, HeaderSpatial, HeaderVersion, NiftiDatatype, NiftiHeader};

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

    let parsed = NiftiHeader::parse(&header.encode().expect("header encodes"))
        .expect("encoded header parses");
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
        NiftiDatatype::Uint32,
        HeaderSpatial {
            pixdim: [1.0, 0.75, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0],
            srow_x: [-0.75, 0.0, 0.0, -11.0],
            srow_y: [0.0, -1.5, 0.0, 7.5],
            srow_z: [0.0, 0.0, 2.0, 3.25],
        },
    )
    .expect("valid header");

    let parsed = NiftiHeader::parse(&header.encode().expect("header encodes"))
        .expect("encoded header parses");
    assert_eq!(parsed.version, HeaderVersion::Two);
    assert_eq!(parsed.dim, [3, 70_000, 3, 2, 1, 1, 1, 1]);
    assert_eq!(parsed.datatype, NiftiDatatype::Uint32);
    assert_eq!(parsed.srow_x, [-0.75, 0.0, 0.0, -11.0]);
    assert_eq!(parsed.vox_offset, 544);
}

#[test]
fn nifti1_round_trips_maximum_positive_dimension() {
    let maximum = usize::try_from(i16::MAX).expect("positive i16 fits usize");
    let header = NiftiHeader::new_volume(
        HeaderDims {
            nx: maximum,
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
    .expect("NIfTI-1 accepts its largest positive signed dimension");

    let parsed = NiftiHeader::parse(&header.encode().expect("header encodes"))
        .expect("signed dimension round-trips");
    assert_eq!(parsed.dim[1], maximum);
}

#[test]
fn nifti1_rejects_dimension_above_positive_signed_range() {
    let first_unrepresentable = usize::try_from(i16::MAX)
        .expect("positive i16 fits usize")
        .saturating_add(1);
    let err = NiftiHeader::new_volume(
        HeaderDims {
            nx: first_unrepresentable,
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
    .expect_err("NIfTI-1 dimensions above its signed field range must be rejected");

    assert!(
        err.to_string().contains("signed 16-bit"),
        "error must name NIfTI-1 dimension bound: {err}"
    );
}

#[test]
fn nifti1_rejects_negative_dimension_fields() {
    let header = NiftiHeader::new_volume(
        HeaderDims {
            nx: 2,
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
    .expect("valid header");
    let mut bytes = header.encode().expect("header encodes");
    bytes[42..44].copy_from_slice(&(-1_i16).to_le_bytes());

    let err = NiftiHeader::parse(&bytes).expect_err("negative dimension is invalid");
    assert!(
        err.to_string().contains("dim[1]") && err.to_string().contains("non-negative"),
        "error must identify the invalid NIfTI-1 dimension: {err}"
    );
}
