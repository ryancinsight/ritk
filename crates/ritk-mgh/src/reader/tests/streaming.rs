use super::*;
use proptest::prelude::*;
use std::io::Cursor;

fn assert_streamed_payload(
    label: &str,
    mri_type: i32,
    payload: &[u8],
    expected: &[f32],
) -> Result<()> {
    let width = i32::try_from(expected.len()).context("test width exceeds i32")?;
    let bytes = build_mgh_bytes(
        VERSION,
        [width, 1, 1],
        SINGLE_FRAME,
        mri_type,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        payload,
    );
    let decoded = decode_mgh(&mut Cursor::new(bytes))?;
    let volumes = decoded.volumes;
    assert_eq!(decoded.dims, [1, 1, expected.len()], "{label} shape");
    assert_eq!(volumes.len(), 1, "{label} single frame");
    assert_eq!(volumes[0].len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in volumes[0].iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label} voxel[{index}]: expected {expected}, got {actual}"
        );
    }
    Ok(())
}

#[test]
fn all_voxel_types_cross_decode_chunk_boundaries_exactly() -> Result<()> {
    let chunk_bytes = voxel_decode::input_chunk_bytes();

    let byte_values: Vec<u8> = (0..chunk_bytes + 3)
        .map(|index| u8::try_from(index % 251).expect("remainder fits u8"))
        .collect();
    let byte_expected: Vec<f32> = byte_values.iter().copied().map(f32::from).collect();
    assert_streamed_payload("MRI_UCHAR", MRI_UCHAR, &byte_values, &byte_expected)?;

    let short_count = chunk_bytes / 2 + 3;
    let short_values: Vec<i16> = (0..short_count)
        .map(|index| {
            i16::try_from(index).expect("test index fits i16")
                - i16::try_from(short_count / 2).expect("test midpoint fits i16")
        })
        .collect();
    let short_payload: Vec<u8> = short_values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect();
    let short_expected: Vec<f32> = short_values.iter().copied().map(f32::from).collect();
    assert_streamed_payload("MRI_SHORT", MRI_SHORT, &short_payload, &short_expected)?;

    let integer_count = chunk_bytes / 4 + 3;
    let integer_values: Vec<i32> = (0..integer_count)
        .map(|index| i32::try_from(index).expect("test index fits i32") * 17 - 34_000)
        .collect();
    let integer_payload: Vec<u8> = integer_values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect();
    let integer_expected: Vec<f32> = integer_values
        .iter()
        .copied()
        .map(|value| value as f32)
        .collect();
    assert_streamed_payload("MRI_INT", MRI_INT, &integer_payload, &integer_expected)?;

    let float_count = chunk_bytes / 4 + 3;
    let float_values: Vec<f32> = (0..float_count)
        .map(|index| index as f32 * 0.125 - 20.0)
        .collect();
    let float_payload: Vec<u8> = float_values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect();
    assert_streamed_payload("MRI_FLOAT", MRI_FLOAT, &float_payload, &float_values)?;

    Ok(())
}

#[test]
fn truncation_after_one_complete_chunk_names_first_incomplete_voxel() {
    let chunk_bytes = voxel_decode::input_chunk_bytes();
    for (label, mri_type, bytes_per_voxel) in [
        ("MRI_UCHAR", MRI_UCHAR, 1usize),
        ("MRI_SHORT", MRI_SHORT, 2usize),
        ("MRI_INT", MRI_INT, 4usize),
        ("MRI_FLOAT", MRI_FLOAT, 4usize),
    ] {
        let chunk_voxels = chunk_bytes / bytes_per_voxel;
        let voxel_count = chunk_voxels + 3;
        let mut payload = vec![0u8; voxel_count * bytes_per_voxel];
        payload.pop();
        let bytes = build_mgh_bytes(
            VERSION,
            [
                i32::try_from(voxel_count).expect("test voxel count fits i32"),
                1,
                1,
            ],
            SINGLE_FRAME,
            mri_type,
            [1.0, 1.0, 1.0],
            IDENTITY_DIR,
            [0.0, 0.0, 0.0],
            &payload,
        );
        let error = match decode_mgh(&mut Cursor::new(bytes)) {
            Ok(_) => panic!("one missing payload byte must reject the volume"),
            Err(error) => error,
        };
        let message = format!("{error:#}");
        let first_incomplete_voxel = chunk_voxels + 2;
        assert!(
            message.contains(&format!("voxel {first_incomplete_voxel}")),
            "{label} error must name the first unconfirmed voxel; got {message}"
        );
    }
}

proptest! {
    #[test]
    fn arbitrary_u8_payload_is_exact_or_rejected(
        nx in 1usize..=16,
        ny in 1usize..=16,
        nz in 1usize..=16,
        payload in proptest::collection::vec(any::<u8>(), 0..=4_200),
    ) {
        let voxel_count = nx * ny * nz;
        let bytes = build_mgh_bytes(
            VERSION,
            [nx as i32, ny as i32, nz as i32],
            SINGLE_FRAME,
            MRI_UCHAR,
            [1.0, 1.0, 1.0],
            IDENTITY_DIR,
            [0.0, 0.0, 0.0],
            &payload,
        );
        let result = decode_mgh(&mut Cursor::new(bytes));
        if payload.len() < voxel_count {
            let error = match result {
                Ok(_) => return Err(TestCaseError::fail("short arbitrary payload must fail")),
                Err(error) => error,
            };
            prop_assert!(
                format!("{error:#}").contains("truncated"),
                "short-payload error must identify truncation"
            );
        } else {
            let decoded = result.expect("complete arbitrary payload must decode");
            prop_assert_eq!(decoded.dims, [nz, ny, nx]);
            prop_assert_eq!(decoded.volumes.len(), 1, "arbitrary payload is single-frame");
            let expected: Vec<f32> = payload[..voxel_count]
                .iter()
                .copied()
                .map(f32::from)
                .collect();
            prop_assert_eq!(&decoded.volumes[0], &expected);
        }
    }
}
