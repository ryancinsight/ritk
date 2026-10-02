use consus_core::{write_integer, ByteOrder, EndianScalar};

use super::{decode_samples, Sample, SampleBuffer, SampleError, SampleType};

/// Encode `values` in `order` through consus-core's scalar writer, an encoder
/// independent of the bulk decoder under test.
fn encode<T: Sample + EndianScalar>(values: &[T], order: ByteOrder) -> Vec<u8> {
    let width = T::TYPE.byte_width();
    let mut bytes = vec![0_u8; values.len() * width];
    for (value, slot) in values.iter().zip(bytes.chunks_exact_mut(width)) {
        write_integer(slot, *value, order).expect("invariant: slot holds exactly one sample");
    }
    bytes
}

/// Zero, one, the type's extremes, and a value exercising every byte.
fn probes<T: Sample>() -> Vec<T> {
    let bytes: Vec<u8> = (1..=T::TYPE.byte_width())
        .map(|b| u8::try_from(b * 0x11).expect("invariant: at most 8 * 0x11 = 0x88"))
        .collect();
    let every_byte = bytemuck::pod_read_unaligned::<T>(&bytes);
    vec![T::zero(), T::one(), every_byte]
}

fn decode_round_trips<T: Sample + EndianScalar + PartialEq + std::fmt::Debug>() {
    let values = probes::<T>();
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        let bytes = encode(&values, order);
        let decoded: Vec<T> = decode_samples(&bytes, order).expect("whole samples");
        assert_eq!(decoded, values, "{} in {order:?}", T::TYPE);

        let buffer = SampleBuffer::decode(&bytes, T::TYPE, order).expect("whole samples");
        assert_eq!(buffer.sample_type(), T::TYPE);
        assert_eq!(buffer.len(), values.len());
        assert_eq!(
            buffer.try_into_vec::<T>().expect("stored type matches"),
            values,
            "{} in {order:?}",
            T::TYPE
        );
    }
}

#[test]
fn decode_round_trips_every_sample_type_in_both_byte_orders() {
    decode_round_trips::<u8>();
    decode_round_trips::<i8>();
    decode_round_trips::<u16>();
    decode_round_trips::<i16>();
    decode_round_trips::<u32>();
    decode_round_trips::<i32>();
    decode_round_trips::<u64>();
    decode_round_trips::<i64>();
    decode_round_trips::<f32>();
    decode_round_trips::<f64>();
}

#[test]
fn decode_reads_literal_byte_orders() {
    let bytes = [0x01, 0x02, 0x03, 0x04];
    assert_eq!(
        decode_samples::<u16>(&bytes, ByteOrder::BigEndian).expect("two samples"),
        [0x0102, 0x0304]
    );
    assert_eq!(
        decode_samples::<u16>(&bytes, ByteOrder::LittleEndian).expect("two samples"),
        [0x0201, 0x0403]
    );
    assert_eq!(
        decode_samples::<i32>(&[0xFF, 0xFF, 0xFF, 0xFE], ByteOrder::BigEndian).expect("one"),
        [-2]
    );
    assert_eq!(
        decode_samples::<f64>(&1.5_f64.to_be_bytes(), ByteOrder::BigEndian).expect("one"),
        [1.5]
    );
}

/// Values an `f32` working type rounds: 2^24 + 1 needs 25 significand bits,
/// 2^53 + 1 needs 54, and the `f64` below needs all 53.
#[test]
fn decode_keeps_values_f32_cannot_represent() {
    let order = ByteOrder::LittleEndian;
    let i32_probe = (1_i32 << 24) + 1;
    let u32_probe = (1_u32 << 24) + 1;
    let i64_probe = -((1_i64 << 53) + 1);
    let u64_probe = u64::MAX;
    let f64_probe = 1.0 + f64::EPSILON;

    let decoded_i32 = SampleBuffer::decode(&encode(&[i32_probe], order), SampleType::I32, order)
        .expect("one sample")
        .try_into_vec::<i32>()
        .expect("stored type matches");
    let decoded_u32 = SampleBuffer::decode(&encode(&[u32_probe], order), SampleType::U32, order)
        .expect("one sample")
        .try_into_vec::<u32>()
        .expect("stored type matches");
    let decoded_i64 = SampleBuffer::decode(&encode(&[i64_probe], order), SampleType::I64, order)
        .expect("one sample")
        .try_into_vec::<i64>()
        .expect("stored type matches");
    let decoded_u64 = SampleBuffer::decode(&encode(&[u64_probe], order), SampleType::U64, order)
        .expect("one sample")
        .try_into_vec::<u64>()
        .expect("stored type matches");
    let decoded_f64 = SampleBuffer::decode(&encode(&[f64_probe], order), SampleType::F64, order)
        .expect("one sample")
        .try_into_vec::<f64>()
        .expect("stored type matches");

    assert_eq!(decoded_i32, [i32_probe]);
    assert_eq!(decoded_u32, [u32_probe]);
    assert_eq!(decoded_i64, [i64_probe]);
    assert_eq!(decoded_u64, [u64_probe]);
    assert_eq!(decoded_f64, [f64_probe]);
}

#[test]
fn exact_conversion_preserves_representable_values() {
    let wide = SampleBuffer::I32(vec![(1 << 24) + 1, i32::MIN])
        .try_convert::<i64>()
        .expect("every i32 is exactly represented by i64");
    assert_eq!(wide, [(1_i64 << 24) + 1, i64::from(i32::MIN)]);
    let wide = SampleBuffer::U16(vec![u16::MAX])
        .try_convert::<f64>()
        .expect("every u16 is exactly represented by f64");
    assert_eq!(wide, [65535.0]);
}

#[test]
fn same_type_conversion_preserves_the_vector_and_float_bits() {
    let samples = vec![f64::from_bits(0x7FF8_0000_0000_1234), -0.0];
    let address = samples.as_ptr();
    let (converted, report) = SampleBuffer::F64(samples)
        .convert_lossy::<f64>()
        .expect("same-type conversion moves the vector")
        .into_parts();

    assert_eq!(converted.as_ptr(), address);
    assert_eq!(
        converted
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [0x7FF8_0000_0000_1234, (-0.0_f64).to_bits()]
    );
    assert_eq!(report.source_type, SampleType::F64);
    assert_eq!(report.target_type, SampleType::F64);
    assert_eq!(report.changed_samples, 0);
    assert_eq!(report.first_changed_sample, None);
}

#[test]
fn exact_integer_inspection_respects_i128_bounds() {
    assert_eq!((-2.0_f64).powi(127).exact_integer_value(), Some(i128::MIN));
    assert_eq!((2.0_f64).powi(127).exact_integer_value(), None);
    assert_eq!((2.0_f64).powi(128).exact_integer_value(), None);
    assert_eq!(f64::MAX.exact_integer_value(), None);
    assert_eq!(1.5_f64.exact_integer_value(), None);
    assert_eq!((-0.0_f64).exact_integer_value(), Some(0));
}

#[test]
fn exact_extraction_returns_mismatched_buffer_unchanged() {
    let returned = SampleBuffer::U16(vec![300])
        .try_into_vec::<u8>()
        .expect_err("a mismatched stored type must not be cast");
    assert_eq!(returned, SampleBuffer::U16(vec![300]));
}

#[test]
fn lossy_conversion_reports_changed_values() {
    let (narrowed, report) = SampleBuffer::U16(vec![300, 255])
        .convert_lossy::<u8>()
        .expect("small output allocation")
        .into_parts();
    assert_eq!(narrowed, [44, 255]);
    assert_eq!(report.source_type, SampleType::U16);
    assert_eq!(report.target_type, SampleType::U8);
    assert_eq!(report.changed_samples, 1);
    assert_eq!(report.first_changed_sample, Some(0));

    let (rounded, report) = SampleBuffer::I32(vec![(1 << 24) + 1])
        .convert_lossy::<f32>()
        .expect("small output allocation")
        .into_parts();
    assert_eq!(rounded, [16_777_216.0]);
    assert_eq!(report.changed_samples, 1);
    assert_eq!(report.first_changed_sample, Some(0));

    let (rounded, report) = SampleBuffer::U64(vec![u64::MAX])
        .convert_lossy::<f32>()
        .expect("small output allocation")
        .into_parts();
    assert_eq!(rounded[0].to_bits(), 0x5F80_0000);
    assert_eq!(report.changed_samples, 1);

    let (reinterpreted, report) = SampleBuffer::U64(vec![u64::MAX])
        .convert_lossy::<i64>()
        .expect("small output allocation")
        .into_parts();
    assert_eq!(reinterpreted, [-1]);
    assert_eq!(report.changed_samples, 1);

    let (_, report) = SampleBuffer::I64(vec![i64::MIN])
        .convert_lossy::<f32>()
        .expect("small output allocation")
        .into_parts();
    assert_eq!(report.changed_samples, 0);

    let (saturated, report) = SampleBuffer::F64(vec![2.0_f64.powi(64)])
        .convert_lossy::<u64>()
        .expect("small output allocation")
        .into_parts();
    assert_eq!(saturated, [u64::MAX]);
    assert_eq!(report.changed_samples, 1);

    let (_, report) = SampleBuffer::F64(vec![(-0.0_f64)])
        .convert_lossy::<i32>()
        .expect("small output allocation")
        .into_parts();
    assert_eq!(report.changed_samples, 1);
}

#[test]
fn exact_conversion_rejects_first_changed_sample() {
    let error = SampleBuffer::I32(vec![1, (1 << 24) + 1, 3])
        .try_convert::<f32>()
        .expect_err("one integer is not exactly represented by f32");
    assert!(matches!(
        error,
        super::SampleConversionError::Changed {
            source_type: SampleType::I32,
            target_type: SampleType::F32,
            first_changed_sample: 1,
            changed_samples: 1,
        }
    ));
}

#[test]
fn decoding_preserves_signed_zero_and_nan_payload_bits() {
    let bytes = (-0.0_f32).to_be_bytes();
    let values = SampleBuffer::decode(&bytes, SampleType::F32, ByteOrder::BigEndian)
        .expect("one f32 sample")
        .try_into_vec::<f32>()
        .expect("stored type matches");
    assert_eq!(values[0].to_bits(), (-0.0_f32).to_bits());

    let bytes = 0x7FC1_2345_u32.to_be_bytes();
    let values = SampleBuffer::decode(&bytes, SampleType::F32, ByteOrder::BigEndian)
        .expect("one f32 sample")
        .try_into_vec::<f32>()
        .expect("stored type matches");
    assert_eq!(values[0].to_bits(), 0x7FC1_2345);

    let bytes = 0x7FF8_0000_0000_1234_u64.to_le_bytes();
    let values = SampleBuffer::decode(&bytes, SampleType::F64, ByteOrder::LittleEndian)
        .expect("one f64 sample")
        .try_into_vec::<f64>()
        .expect("stored type matches");
    assert_eq!(values[0].to_bits(), 0x7FF8_0000_0000_1234);
}

fn converts_from_each_sample_type<S: Sample>() {
    let (converted, report) = S::into_buffer(vec![S::zero(), S::one()])
        .convert_lossy::<f64>()
        .expect("small output allocation")
        .into_parts();
    assert_eq!(converted, [0.0, 1.0]);
    assert_eq!(report.source_type, S::TYPE);
    assert_eq!(report.target_type, SampleType::F64);
    assert_eq!(report.changed_samples, 0);
}

fn converts_to_each_sample_type<T: Sample>() {
    let (converted, report) = SampleBuffer::U8(vec![0, 1])
        .convert_lossy::<T>()
        .expect("small output allocation")
        .into_parts();
    assert_eq!(
        converted
            .into_iter()
            .map(|value| value.exact_integer_value())
            .collect::<Vec<_>>(),
        [Some(0), Some(1)]
    );
    assert_eq!(report.source_type, SampleType::U8);
    assert_eq!(report.target_type, T::TYPE);
    assert_eq!(report.changed_samples, 0);
}

#[test]
fn conversion_dispatch_covers_every_stored_primitive() {
    converts_from_each_sample_type::<u8>();
    converts_from_each_sample_type::<i8>();
    converts_from_each_sample_type::<u16>();
    converts_from_each_sample_type::<i16>();
    converts_from_each_sample_type::<u32>();
    converts_from_each_sample_type::<i32>();
    converts_from_each_sample_type::<u64>();
    converts_from_each_sample_type::<i64>();
    converts_from_each_sample_type::<f32>();
    converts_from_each_sample_type::<f64>();

    converts_to_each_sample_type::<u8>();
    converts_to_each_sample_type::<i8>();
    converts_to_each_sample_type::<u16>();
    converts_to_each_sample_type::<i16>();
    converts_to_each_sample_type::<u32>();
    converts_to_each_sample_type::<i32>();
    converts_to_each_sample_type::<u64>();
    converts_to_each_sample_type::<i64>();
    converts_to_each_sample_type::<f32>();
    converts_to_each_sample_type::<f64>();
}

#[test]
fn decode_rejects_a_partial_trailing_sample() {
    for sample_type in SampleType::ALL {
        let width = sample_type.byte_width();
        if width == 1 {
            continue;
        }
        let bytes = vec![0_u8; width + 1];
        let err = SampleBuffer::decode(&bytes, sample_type, ByteOrder::LittleEndian)
            .expect_err("a partial sample must be rejected");
        assert!(matches!(
            err,
            SampleError::PartialSample {
                sample_type: found_type,
                byte_len,
            } if found_type == sample_type && byte_len == width + 1
        ));
    }
}

#[test]
fn decode_of_no_bytes_is_empty() {
    for sample_type in SampleType::ALL {
        let buffer = SampleBuffer::decode(&[], sample_type, ByteOrder::BigEndian)
            .expect("zero samples decode");
        assert!(buffer.is_empty());
        assert_eq!(buffer.sample_type(), sample_type);
    }
}

#[test]
fn sample_type_widths_match_the_primitives() {
    let widths = SampleType::ALL.map(SampleType::byte_width);
    assert_eq!(widths, [1, 1, 2, 2, 4, 4, 8, 8, 4, 8]);
    let floats: Vec<SampleType> = SampleType::ALL
        .into_iter()
        .filter(|t| t.is_float())
        .collect();
    assert_eq!(floats, [SampleType::F32, SampleType::F64]);
    assert_eq!(SampleType::I64.to_string(), "i64");

    assert_sample_width::<u8>();
    assert_sample_width::<i8>();
    assert_sample_width::<u16>();
    assert_sample_width::<i16>();
    assert_sample_width::<u32>();
    assert_sample_width::<i32>();
    assert_sample_width::<u64>();
    assert_sample_width::<i64>();
    assert_sample_width::<f32>();
    assert_sample_width::<f64>();
}

fn assert_sample_width<T: Sample>() {
    assert_eq!(T::TYPE.byte_width(), std::mem::size_of::<T>());
}
