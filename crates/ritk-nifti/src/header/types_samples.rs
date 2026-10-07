use super::{HeaderDims, HeaderSpatial, NiftiDatatype, NiftiHeader};
use anyhow::Result;
use ritk_codecs::SampleType;

#[test]
fn one_byte_voxel_readers_reject_short_and_oversized_lanes() -> Result<()> {
    let header = NiftiHeader::new_volume(
        HeaderDims {
            nx: 1,
            ny: 1,
            nz: 1,
        },
        NiftiDatatype::Uint8,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
        },
    )?;

    for raw in [&[][..], &[1, 2][..]] {
        let float_error = header
            .read_f32_voxel(raw)
            .expect_err("wrong-width voxel lane must fail");
        assert!(
            float_error
                .to_string()
                .contains("voxel lane width mismatch"),
            "float reader reports lane width: {float_error}"
        );

        let label_error = header
            .read_label_voxel(raw)
            .expect_err("wrong-width label lane must fail");
        assert!(
            label_error
                .to_string()
                .contains("voxel lane width mismatch"),
            "label reader reports lane width: {label_error}"
        );
    }
    Ok(())
}

#[test]
fn every_stored_sample_type_decodes_big_endian_voxel_bytes() -> Result<()> {
    let samples = [
        (SampleType::U8, vec![0x23], 35.0_f32),
        (SampleType::I8, (-8_i8).to_be_bytes().to_vec(), -8.0),
        (SampleType::U16, 0x1203_u16.to_be_bytes().to_vec(), 4611.0),
        (SampleType::I16, (-2010_i16).to_be_bytes().to_vec(), -2010.0),
        (
            SampleType::U32,
            16_777_217_u32.to_be_bytes().to_vec(),
            16_777_216.0,
        ),
        (
            SampleType::I32,
            (-16_777_217_i32).to_be_bytes().to_vec(),
            -16_777_216.0,
        ),
        (
            SampleType::U64,
            4_294_967_297_u64.to_be_bytes().to_vec(),
            4_294_967_296.0,
        ),
        (
            SampleType::I64,
            (-4_294_967_297_i64).to_be_bytes().to_vec(),
            -4_294_967_296.0,
        ),
        (SampleType::F32, 1.25_f32.to_be_bytes().to_vec(), 1.25),
        (SampleType::F64, (-4.5_f64).to_be_bytes().to_vec(), -4.5),
    ];

    for (sample_type, sample_bytes, expected) in samples {
        let mut header = NiftiHeader::new_volume(
            HeaderDims {
                nx: 1,
                ny: 1,
                nz: 1,
            },
            NiftiDatatype::try_from(sample_type).expect("supported sample type"),
            HeaderSpatial {
                pixdim: [1.0; 8],
                srow_x: [1.0, 0.0, 0.0, 0.0],
                srow_y: [0.0, 1.0, 0.0, 0.0],
                srow_z: [0.0, 0.0, 1.0, 0.0],
            },
        )?;
        header.endian = consus_core::ByteOrder::BigEndian;
        assert_eq!(header.read_f32_voxel(&sample_bytes)?, expected);
    }
    Ok(())
}
