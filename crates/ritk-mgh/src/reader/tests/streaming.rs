use super::*;
use proptest::prelude::*;
use ritk_codecs::sample::Sample;
use ritk_core::alloc_probe::peak_bytes_during;
use std::io::Cursor;

/// consus-core's `read_extend` step: the payloads below run past it so the
/// step boundary is crossed with a partial final step.
const STREAM_STEP_BYTES: usize = 16 * 1024;

/// A single-row MGH stream of `values` stored as `code`.
fn stream_of<T: Sample>(code: i32, values: &[T]) -> Vec<u8> {
    let mut payload = Vec::new();
    ritk_codecs::sample::write_samples(values, consus_core::ByteOrder::BigEndian, &mut payload)
        .expect("a vector accepts every byte");
    let width = i32::try_from(values.len()).expect("invariant: test width fits i32");
    build_mgh_bytes(
        VERSION,
        [width, 1, 1],
        SINGLE_FRAME,
        code,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &payload,
    )
}

/// `values` read back from their MGH stream in their own type.
fn assert_streamed_payload<T: Sample + std::fmt::Debug>(code: i32, values: &[T]) -> Result<()> {
    let decoded = decode_mgh::<T, _, _>(&mut Cursor::new(stream_of(code, values)), Exact)?;
    assert_eq!(decoded.dims, [1, 1, values.len()], "{}", T::TYPE);
    assert_eq!(decoded.volumes, [values], "{}", T::TYPE);
    Ok(())
}

/// `count` values `value(index)`.
fn ramp<T>(count: usize, value: impl Fn(i64) -> T) -> Vec<T> {
    (0..count)
        .map(|index| value(i64::try_from(index).expect("invariant: test index fits i64")))
        .collect()
}

#[test]
fn all_voxel_types_cross_stream_steps_exactly() -> Result<()> {
    assert_streamed_payload(
        MRI_UCHAR,
        &ramp(STREAM_STEP_BYTES + 3, |i| {
            u8::try_from(i % 251).expect("below 251")
        }),
    )?;
    assert_streamed_payload(
        MRI_SHORT,
        &ramp(STREAM_STEP_BYTES / 2 + 3, |i| {
            i16::try_from(i - 4_000).expect("invariant: small ramp")
        }),
    )?;
    // Values past 2^24 have no exact f32: the stored i32 must survive.
    assert_streamed_payload(
        MRI_INT,
        &ramp(STREAM_STEP_BYTES / 4 + 3, |i| {
            i32::try_from(i * 4_099 - 16_777_217).expect("invariant: ramp fits i32")
        }),
    )?;
    let floats = ramp(STREAM_STEP_BYTES / 4 + 3, |i| {
        f32::from(i16::try_from(i - 160).expect("invariant: small ramp")) * 0.125
    });
    assert_streamed_payload(MRI_FLOAT, &floats)
}

#[test]
fn truncation_after_one_complete_step_names_first_incomplete_voxel() {
    for (mri_type, bytes_per_voxel) in
        [(MRI_UCHAR, 1), (MRI_SHORT, 2), (MRI_INT, 4), (MRI_FLOAT, 4)]
    {
        let step_voxels = STREAM_STEP_BYTES / bytes_per_voxel;
        let voxel_count = step_voxels + 3;
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
        let error = match decode_mgh::<f64, _, _>(&mut Cursor::new(bytes), Exact) {
            Ok(_) => panic!("one missing payload byte must reject the volume"),
            Err(error) => error,
        };
        let message = format!("{error:#}");
        let first_incomplete_voxel = step_voxels + 2;
        assert!(
            message.contains("truncated")
                && message.contains(&format!("sample {first_incomplete_voxel} of {voxel_count}")),
            "type {mri_type}: the error must name the first unconfirmed voxel; got {message}"
        );
    }
}

/// A header declaring `i32::MAX` frames must not reserve for them before the
/// payload proves they exist.
///
/// The honest cost is the header scratch, one voxel, and the error chain,
/// measured at 4.8 KiB; the bound is one 16 KiB `read_extend` step. The defect
/// reserved `i32::MAX` frame slots up front, measured at 48 GiB, six orders of
/// magnitude above the bound, so the threshold has no near miss either way.
#[test]
fn a_hostile_frame_count_reserves_nothing_before_its_payload() {
    const HONEST_PEAK_BOUND: usize = 16 * 1024;
    let bytes = build_mgh_bytes(
        VERSION,
        [1, 1, 1],
        i32::MAX,
        MRI_UCHAR,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &[7],
    );
    let mut stream = Cursor::new(bytes);
    let (result, peak) = peak_bytes_during(|| decode_mgh::<u8, _, _>(&mut stream, Exact));
    let error = match result {
        Ok(_) => panic!("one payload byte cannot supply i32::MAX frames"),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("Failed to decode MGH frame 1 of 2147483647")
            && message.contains("MGH voxel payload is truncated"),
        "{message}"
    );
    assert!(
        peak < HONEST_PEAK_BOUND,
        "peak live bytes {peak} reach the {HONEST_PEAK_BOUND}-byte honest bound"
    );
}

proptest! {
    #[test]
    fn arbitrary_byte_payload_is_exact_or_rejected(
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
        let result = decode_mgh::<u8, _, _>(&mut Cursor::new(bytes), Exact);
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
            prop_assert_eq!(&decoded.volumes[0][..], &payload[..voxel_count]);
        }
    }
}
