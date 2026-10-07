use super::*;
use proptest::prelude::*;

#[test]
fn image_reader_converts_all_stored_scalar_types_at_the_f32_boundary() -> Result<()> {
    use ritk_codecs::SampleType;

    fn little_endian<T, const N: usize>(
        values: impl IntoIterator<Item = T>,
        encode: fn(T) -> [u8; N],
    ) -> Vec<u8> {
        values.into_iter().flat_map(encode).collect()
    }

    let samples = [
        (SampleType::U8, vec![3, 241], vec![3.0_f32, 241.0]),
        (
            SampleType::I8,
            little_endian([-5_i8, 100], i8::to_le_bytes),
            vec![-5.0, 100.0],
        ),
        (
            SampleType::U16,
            little_endian([513_u16, 65_530], u16::to_le_bytes),
            vec![513.0, 65_530.0],
        ),
        (
            SampleType::I16,
            little_endian([-1026_i16, 32_760], i16::to_le_bytes),
            vec![-1026.0, 32_760.0],
        ),
        (
            SampleType::U32,
            little_endian([16_777_217_u32, 70_000], u32::to_le_bytes),
            vec![16_777_216.0, 70_000.0],
        ),
        (
            SampleType::I32,
            little_endian([-16_777_217_i32, -70_000], i32::to_le_bytes),
            vec![-16_777_216.0, -70_000.0],
        ),
        (
            SampleType::U64,
            little_endian([4_294_967_297_u64, 16_777_217], u64::to_le_bytes),
            vec![4_294_967_296.0, 16_777_216.0],
        ),
        (
            SampleType::I64,
            little_endian([-4_294_967_297_i64, -16_777_217], i64::to_le_bytes),
            vec![-4_294_967_296.0, -16_777_216.0],
        ),
        (
            SampleType::F32,
            little_endian([7.25_f32, -11.5], f32::to_le_bytes),
            vec![7.25, -11.5],
        ),
        (
            SampleType::F64,
            little_endian([1.5_f64, -3.125], f64::to_le_bytes),
            vec![1.5, -3.125],
        ),
    ];
    let backend = SequentialBackend;

    for (sample_type, sample_bytes, expected) in samples {
        let datatype = NiftiDatatype::try_from(sample_type).expect("supported NIfTI sample type");
        for version in [HeaderVersion::One, HeaderVersion::Two] {
            let header = NiftiHeader::new_with_version(
                version,
                HeaderDims {
                    nx: expected.len(),
                    ny: 1,
                    nz: 1,
                },
                1,
                datatype,
                HeaderSpatial {
                    pixdim: [1.0; 8],
                    srow_x: [1.0, 0.0, 0.0, 0.0],
                    srow_y: [0.0, 1.0, 0.0, 0.0],
                    srow_z: [0.0, 0.0, 1.0, 0.0],
                },
            )?;
            let bytes = write_single_file_bytes(&header, &sample_bytes);
            let image = read_nifti_from_bytes(&bytes, &backend)?;

            assert_eq!(image.shape(), [1, 1, expected.len()], "{version:?}");
            assert_eq!(
                image.data_slice().expect("contiguous image"),
                expected,
                "{version:?} {sample_type:?}"
            );
        }
    }

    Ok(())
}

#[test]
fn image_reader_rejects_unrepresentable_spacing_without_panicking() -> Result<()> {
    let mut header = NiftiHeader::new_with_version(
        HeaderVersion::Two,
        HeaderDims {
            nx: 1,
            ny: 1,
            nz: 1,
        },
        1,
        NiftiDatatype::Float64,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
        },
    )?;
    header.qform_code = 0;
    header.sform_code = 0;
    header.pixdim[1] = 1.0e100;
    let bytes = write_single_file_bytes(&header, &1.5_f64.to_le_bytes());

    let error = read_nifti_from_bytes(&bytes, &SequentialBackend)
        .expect_err("spacing outside f32 range must return a typed error");
    assert!(
        format!("{error:#}").contains("f32-representable"),
        "error identifies the image spacing boundary: {error:#}"
    );
    Ok(())
}

#[test]
fn image_reader_rejects_finite_f64_samples_outside_f32_range() -> Result<()> {
    let header = NiftiHeader::new_with_version(
        HeaderVersion::Two,
        HeaderDims {
            nx: 1,
            ny: 1,
            nz: 1,
        },
        1,
        NiftiDatatype::Float64,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
        },
    )?;
    let bytes = write_single_file_bytes(&header, &f64::MAX.to_le_bytes());

    let error = read_nifti_from_bytes(&bytes, &SequentialBackend)
        .expect_err("finite f64 sample outside f32 range must fail");
    assert!(
        format!("{error:#}").contains("f32-representable"),
        "error identifies the f32 image boundary: {error:#}"
    );
    Ok(())
}

#[test]
fn image_reader_preserves_nonfinite_f64_sample_categories() -> Result<()> {
    let header = NiftiHeader::new_with_version(
        HeaderVersion::Two,
        HeaderDims {
            nx: 3,
            ny: 1,
            nz: 1,
        },
        1,
        NiftiDatatype::Float64,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
        },
    )?;
    let samples = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
        .into_iter()
        .flat_map(f64::to_le_bytes)
        .collect::<Vec<_>>();
    let bytes = write_single_file_bytes(&header, &samples);
    let image = read_nifti_from_bytes(&bytes, &SequentialBackend)?;
    let values = image.data_slice().expect("contiguous image");

    assert!(values[0].is_nan());
    assert_eq!(values[1], f32::INFINITY);
    assert_eq!(values[2], f32::NEG_INFINITY);
    Ok(())
}

proptest! {
    #[test]
    fn structured_nifti2_header_mutations_never_panic(
        field in 0_u8..18,
        dimension in any::<i64>(),
        vox_offset in any::<i64>(),
        datatype in any::<i16>(),
        bitpix in any::<i16>(),
        form_code in any::<i32>(),
        scalar_bits in any::<u64>(),
    ) {
        fn replace_field<const N: usize>(bytes: &mut [u8], offset: usize, value: [u8; N]) {
            let end = offset.checked_add(N).expect("fixed header field offset");
            bytes
                .get_mut(offset..end)
                .expect("field is within the NIfTI-2 header")
                .copy_from_slice(&value);
        }

        let header = NiftiHeader::new_with_version(
            HeaderVersion::Two,
            HeaderDims { nx: 1, ny: 1, nz: 1 },
            1,
            NiftiDatatype::Float64,
            HeaderSpatial {
                pixdim: [1.0; 8],
                srow_x: [1.0, 0.0, 0.0, 0.0],
                srow_y: [0.0, 1.0, 0.0, 0.0],
                srow_z: [0.0, 0.0, 1.0, 0.0],
            },
        ).expect("fixed NIfTI-2 header is valid");
        let mut header = header;
        header.qform_code = 1;
        header.sform_code = 1;
        let mut bytes = write_single_file_bytes(&header, &1.5_f64.to_le_bytes());
        match field {
            0 => replace_field(&mut bytes, 16, dimension.to_le_bytes()),
            1 => replace_field(&mut bytes, 24, dimension.to_le_bytes()),
            2 => replace_field(&mut bytes, 168, vox_offset.to_le_bytes()),
            3 => replace_field(&mut bytes, 12, datatype.to_le_bytes()),
            4 => replace_field(&mut bytes, 14, bitpix.to_le_bytes()),
            5 => replace_field(&mut bytes, 344, form_code.to_le_bytes()),
            6 => replace_field(&mut bytes, 348, form_code.to_le_bytes()),
            7 => replace_field(&mut bytes, 112, f64::from_bits(scalar_bits).to_le_bytes()),
            8 => replace_field(&mut bytes, 352, f64::from_bits(scalar_bits).to_le_bytes()),
            9 => replace_field(&mut bytes, 400, f64::from_bits(scalar_bits).to_le_bytes()),
            10 => replace_field(&mut bytes, 104, f64::from_bits(scalar_bits).to_le_bytes()),
            11 => replace_field(&mut bytes, 360, f64::from_bits(scalar_bits).to_le_bytes()),
            12 => replace_field(&mut bytes, 368, f64::from_bits(scalar_bits).to_le_bytes()),
            13 => replace_field(&mut bytes, 376, f64::from_bits(scalar_bits).to_le_bytes()),
            14 => replace_field(&mut bytes, 384, f64::from_bits(scalar_bits).to_le_bytes()),
            15 => replace_field(&mut bytes, 392, f64::from_bits(scalar_bits).to_le_bytes()),
            16 => replace_field(&mut bytes, 432, f64::from_bits(scalar_bits).to_le_bytes()),
            17 => replace_field(&mut bytes, 464, f64::from_bits(scalar_bits).to_le_bytes()),
            _ => return Err(TestCaseError::fail("mutation selector is out of range")),
        }

        let document = std::panic::catch_unwind(|| crate::NiftiDocument::from_bytes(&bytes));
        prop_assert!(document.is_ok(), "structured NIfTI document metadata caused a panic");
        if let Ok(Ok(document)) = document {
            prop_assert_eq!(document.sample_bytes(), &1.5_f64.to_le_bytes());
        }

        let image = std::panic::catch_unwind(|| read_nifti_from_bytes(&bytes, &SequentialBackend));
        prop_assert!(image.is_ok(), "structured NIfTI image metadata caused a panic");
        if let Ok(Ok(image)) = image {
            prop_assert_eq!(image.shape(), [1, 1, 1]);
        }
    }

    #[test]
    fn arbitrary_nifti_bytes_do_not_panic(bytes in proptest::collection::vec(any::<u8>(), 0..=1024)) {
        let document = std::panic::catch_unwind(|| crate::NiftiDocument::from_bytes(&bytes));
        prop_assert!(document.is_ok(), "arbitrary NIfTI document bytes caused a panic");

        let image = std::panic::catch_unwind(|| read_nifti_from_bytes(&bytes, &SequentialBackend));
        prop_assert!(image.is_ok(), "arbitrary NIfTI image bytes caused a panic");
    }
}
