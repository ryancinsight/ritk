use std::fmt::Debug;
use std::io::{self, Write};

use crate::ByteOrder;

use super::{Sample, SampleBuffer, SampleError, SampleType};

fn assert_endian_round_trip<T>(samples: Vec<T>)
where
    T: Sample + Debug + PartialEq,
{
    let buffer = SampleBuffer::from_samples(samples.clone());
    assert_eq!(buffer.sample_type(), T::SAMPLE_TYPE);
    assert_eq!(buffer.len(), samples.len());

    for byte_order in [
        ByteOrder::LeastSignificantByteFirst,
        ByteOrder::MostSignificantByteFirst,
    ] {
        let encoded = buffer.encode(byte_order).expect("sample encoding");
        let mut streamed = Vec::new();
        buffer
            .write_to(&mut streamed, byte_order)
            .expect("streamed sample encoding");
        assert_eq!(streamed, encoded);
        let decoded =
            SampleBuffer::decode(T::SAMPLE_TYPE, &encoded, byte_order).expect("sample decoding");
        assert_eq!(
            decoded
                .try_into_samples::<T>()
                .expect("matching sample type"),
            samples
        );
    }
}

#[test]
fn every_fixed_width_sample_round_trips_in_both_byte_orders() {
    assert_endian_round_trip(vec![u8::MIN, 0, u8::MAX]);
    assert_endian_round_trip(vec![i8::MIN, -1, 0, i8::MAX]);
    assert_endian_round_trip(vec![u16::MIN, 1, u16::MAX]);
    assert_endian_round_trip(vec![i16::MIN, -1, 0, i16::MAX]);
    assert_endian_round_trip(vec![0_u32, 16_777_217, u32::MAX]);
    assert_endian_round_trip(vec![i32::MIN, -1, 0, i32::MAX]);
    assert_endian_round_trip(vec![0_u64, 9_007_199_254_740_993, u64::MAX]);
    assert_endian_round_trip(vec![i64::MIN, -1, 0, i64::MAX]);
    assert_endian_round_trip(vec![f32::MIN, -1.5, 0.0, f32::MAX]);
    assert_endian_round_trip(vec![f64::MIN, -1.5, 0.0, f64::MAX]);
}

#[test]
fn endian_encoding_matches_fixed_wire_bytes() {
    fn assert_wire_bytes<T>(sample: T, little_endian: &[u8], big_endian: &[u8])
    where
        T: Sample + Debug + PartialEq,
    {
        let samples = SampleBuffer::from_samples(vec![sample]);
        assert_eq!(
            samples
                .encode(ByteOrder::LeastSignificantByteFirst)
                .expect("little-endian encoding"),
            little_endian
        );
        assert_eq!(
            samples
                .encode(ByteOrder::MostSignificantByteFirst)
                .expect("big-endian encoding"),
            big_endian
        );
        assert_eq!(
            SampleBuffer::decode(
                T::SAMPLE_TYPE,
                little_endian,
                ByteOrder::LeastSignificantByteFirst,
            )
            .expect("little-endian decoding")
            .try_into_samples::<T>()
            .expect("matching sample type"),
            [sample]
        );
        assert_eq!(
            SampleBuffer::decode(
                T::SAMPLE_TYPE,
                big_endian,
                ByteOrder::MostSignificantByteFirst,
            )
            .expect("big-endian decoding")
            .try_into_samples::<T>()
            .expect("matching sample type"),
            [sample]
        );
    }

    assert_wire_bytes(0xa5_u8, &[0xa5], &[0xa5]);
    assert_wire_bytes(-2_i8, &[0xfe], &[0xfe]);
    assert_wire_bytes(0x1234_u16, &[0x34, 0x12], &[0x12, 0x34]);
    assert_wire_bytes(-2_i16, &[0xfe, 0xff], &[0xff, 0xfe]);
    assert_wire_bytes(
        0x0102_0304_u32,
        &[0x04, 0x03, 0x02, 0x01],
        &[0x01, 0x02, 0x03, 0x04],
    );
    assert_wire_bytes(-2_i32, &[0xfe, 0xff, 0xff, 0xff], &[0xff, 0xff, 0xff, 0xfe]);
    assert_wire_bytes(
        0x0102_0304_0506_0708_u64,
        &[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01],
        &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
    );
    assert_wire_bytes(
        -2_i64,
        &[0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe],
    );
    assert_wire_bytes(
        1.0_f32,
        &[0x00, 0x00, 0x80, 0x3f],
        &[0x3f, 0x80, 0x00, 0x00],
    );
    assert_wire_bytes(
        1.0_f64,
        &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f],
        &[0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    );
}

#[test]
fn floating_sample_encoding_preserves_signed_zero_and_nan_payload_bits() {
    let f32_bits = [0x0000_0000, 0x8000_0000, 0x7f80_0001, 0x7fc1_2345];
    let f64_bits = [
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0x7ff8_1234_5678_9abc,
    ];

    for byte_order in [
        ByteOrder::LeastSignificantByteFirst,
        ByteOrder::MostSignificantByteFirst,
    ] {
        let f32_values = f32_bits.map(f32::from_bits).to_vec();
        let f32_buffer = SampleBuffer::from_samples(f32_values);
        let f32_bytes = f32_buffer.encode(byte_order).expect("f32 encoding");
        let mut streamed_f32 = Vec::new();
        f32_buffer
            .write_to(&mut streamed_f32, byte_order)
            .expect("streamed f32 encoding");
        assert_eq!(streamed_f32, f32_bytes);
        let decoded_f32 = SampleBuffer::decode(SampleType::F32, &f32_bytes, byte_order)
            .expect("f32 decoding")
            .try_into_samples::<f32>()
            .expect("f32 sample type");
        assert_eq!(
            decoded_f32
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            f32_bits
        );

        let f64_values = f64_bits.map(f64::from_bits).to_vec();
        let f64_buffer = SampleBuffer::from_samples(f64_values);
        let f64_bytes = f64_buffer.encode(byte_order).expect("f64 encoding");
        let mut streamed_f64 = Vec::new();
        f64_buffer
            .write_to(&mut streamed_f64, byte_order)
            .expect("streamed f64 encoding");
        assert_eq!(streamed_f64, f64_bytes);
        let decoded_f64 = SampleBuffer::decode(SampleType::F64, &f64_bytes, byte_order)
            .expect("f64 decoding")
            .try_into_samples::<f64>()
            .expect("f64 sample type");
        assert_eq!(
            decoded_f64
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            f64_bits
        );
    }
}

#[derive(Debug)]
struct FailingWriter {
    accepted: Vec<u8>,
    limit: usize,
}

impl Write for FailingWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let remaining = self.limit.saturating_sub(self.accepted.len());
        if remaining == 0 {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed output"));
        }
        let accepted = remaining.min(bytes.len());
        self.accepted.extend_from_slice(&bytes[..accepted]);
        Ok(accepted)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn stream_write_preserves_the_written_prefix_and_reports_io_failure() {
    let samples = SampleBuffer::from_samples(vec![0x1234_u16]);
    let mut writer = FailingWriter {
        accepted: Vec::new(),
        limit: 1,
    };

    let error = samples
        .write_to(&mut writer, ByteOrder::LeastSignificantByteFirst)
        .expect_err("a failed output stream must be reported");

    let source = std::error::Error::source(&error).expect("I/O error source is retained");
    assert_eq!(
        source.downcast_ref::<io::Error>().map(io::Error::kind),
        Some(io::ErrorKind::BrokenPipe)
    );
    assert!(matches!(
        error,
        SampleError::Io(error) if error.kind() == io::ErrorKind::BrokenPipe
    ));
    assert_eq!(writer.accepted, [0x34]);
}

#[test]
fn empty_payload_keeps_each_declared_sample_type() {
    for sample_type in SampleType::ALL {
        let decoded = SampleBuffer::decode(sample_type, &[], ByteOrder::LeastSignificantByteFirst)
            .expect("empty payload has no partial sample");
        assert_eq!(decoded.sample_type(), sample_type);
        assert!(decoded.is_empty());
        assert!(decoded
            .encode(ByteOrder::MostSignificantByteFirst)
            .expect("empty samples encode to no bytes")
            .is_empty());
    }
}

#[test]
fn partial_sample_payload_is_rejected_with_its_remainder() {
    for sample_type in SampleType::ALL {
        let sample_width = sample_type.byte_width();
        for byte_order in [
            ByteOrder::LeastSignificantByteFirst,
            ByteOrder::MostSignificantByteFirst,
        ] {
            for complete_samples in 0..=1 {
                for trailing_bytes in 1..sample_width {
                    let byte_length = complete_samples * sample_width + trailing_bytes;
                    let bytes = vec![0; byte_length];
                    let error = SampleBuffer::decode(sample_type, &bytes, byte_order)
                        .expect_err("partial sample must be rejected");

                    assert!(matches!(
                        error,
                        SampleError::PartialSample {
                            sample_type: actual_type,
                            byte_length: actual_length,
                            trailing_bytes: actual_remainder,
                        } if actual_type == sample_type
                            && actual_length == byte_length
                            && actual_remainder == trailing_bytes
                    ));
                }
            }
        }
    }
}

#[test]
fn mismatched_extraction_returns_the_original_samples() {
    let original = SampleBuffer::from_samples(vec![0_u64, 9_007_199_254_740_993, u64::MAX]);
    let original_bytes = original
        .encode(ByteOrder::MostSignificantByteFirst)
        .expect("original encoding");

    let error = original
        .try_into_samples::<u32>()
        .expect_err("different sample types cannot be extracted");
    assert_eq!(error.requested_type(), SampleType::U32);
    assert_eq!(error.actual_type(), SampleType::U64);

    let recovered = error.into_buffer();
    assert_eq!(
        recovered
            .encode(ByteOrder::MostSignificantByteFirst)
            .expect("recovered encoding"),
        original_bytes
    );
    assert_eq!(
        recovered
            .try_into_samples::<u64>()
            .expect("recovered buffer keeps its u64 samples"),
        vec![0, 9_007_199_254_740_993, u64::MAX]
    );
}
