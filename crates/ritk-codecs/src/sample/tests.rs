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
    vec![T::zero(), T::one(), every_byte, T::cast_from_i64(-1)]
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
        assert_eq!(buffer.into_vec::<T>(), values, "{} in {order:?}", T::TYPE);
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
        .into_vec::<i32>();
    let decoded_u32 = SampleBuffer::decode(&encode(&[u32_probe], order), SampleType::U32, order)
        .expect("one sample")
        .into_vec::<u32>();
    let decoded_i64 = SampleBuffer::decode(&encode(&[i64_probe], order), SampleType::I64, order)
        .expect("one sample")
        .into_vec::<i64>();
    let decoded_u64 = SampleBuffer::decode(&encode(&[u64_probe], order), SampleType::U64, order)
        .expect("one sample")
        .into_vec::<u64>();
    let decoded_f64 = SampleBuffer::decode(&encode(&[f64_probe], order), SampleType::F64, order)
        .expect("one sample")
        .into_vec::<f64>();

    assert_eq!(decoded_i32, [i32_probe]);
    assert_eq!(decoded_u32, [u32_probe]);
    assert_eq!(decoded_i64, [i64_probe]);
    assert_eq!(decoded_u64, [u64_probe]);
    assert_eq!(decoded_f64, [f64_probe]);
}

#[test]
fn into_vec_widens_integers_exactly() {
    let wide = SampleBuffer::I32(vec![(1 << 24) + 1, i32::MIN]).into_vec::<i64>();
    assert_eq!(wide, [(1_i64 << 24) + 1, i64::from(i32::MIN)]);
    let wide = SampleBuffer::U16(vec![u16::MAX]).into_vec::<f64>();
    assert_eq!(wide, [65535.0]);
}

/// Cross-type conversion follows the primitive cast: narrowing integers
/// truncate, float-to-integer rounds toward zero and saturates with NaN at
/// zero, and an integer the target float cannot hold rounds to nearest.
#[test]
fn into_vec_applies_the_primitive_cast() {
    assert_eq!(SampleBuffer::I8(vec![-1]).into_vec::<u8>(), [255]);
    assert_eq!(SampleBuffer::U16(vec![300]).into_vec::<u8>(), [44]);
    assert_eq!(
        SampleBuffer::F32(vec![-1.5, 2.9]).into_vec::<i32>(),
        [-1, 2]
    );
    assert_eq!(
        SampleBuffer::F64(vec![f64::NAN, 1e10, -1e10]).into_vec::<i32>(),
        [0, i32::MAX, i32::MIN]
    );
    assert_eq!(
        SampleBuffer::I32(vec![(1 << 24) + 1]).into_vec::<f32>(),
        [16_777_216.0]
    );
    assert_eq!(
        SampleBuffer::F64(vec![1.0 + f64::EPSILON]).into_vec::<f32>(),
        [1.0]
    );
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
        assert_eq!(
            err,
            SampleError::PartialSample {
                sample_type,
                byte_len: width + 1
            }
        );
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
}
