use consus_core::ByteOrder;

use super::{encode, probes};
use crate::sample::{Sample, SampleBuffer, SampleError, SampleType};

fn decode_round_trips<T: Sample + PartialEq + std::fmt::Debug>() {
    assert_eq!(T::TYPE.byte_width(), std::mem::size_of::<T>());
    let values = probes::<T>();
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        let buffer =
            SampleBuffer::decode(&encode(&values, order), T::TYPE, order).expect("whole samples");
        assert_eq!(buffer.sample_type(), T::TYPE);
        assert_eq!(buffer.len(), values.len());
        assert_eq!(
            buffer.into_vec::<T>().expect("the stored type"),
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
    let decode = |sample_type, order| {
        SampleBuffer::decode(&bytes, sample_type, order).expect("whole samples")
    };
    assert_eq!(
        decode(SampleType::U16, ByteOrder::BigEndian),
        SampleBuffer::U16(vec![0x0102, 0x0304])
    );
    assert_eq!(
        decode(SampleType::U16, ByteOrder::LittleEndian),
        SampleBuffer::U16(vec![0x0201, 0x0403])
    );
    assert_eq!(
        decode(SampleType::I32, ByteOrder::BigEndian),
        SampleBuffer::I32(vec![0x0102_0304])
    );
    assert_eq!(
        SampleBuffer::decode(
            &1.5_f64.to_be_bytes(),
            SampleType::F64,
            ByteOrder::BigEndian
        )
        .expect("one sample"),
        SampleBuffer::F64(vec![1.5])
    );
    assert_eq!(
        SampleBuffer::decode(&[0x80, 0xFF], SampleType::I8, ByteOrder::BigEndian)
            .expect("two samples"),
        SampleBuffer::I8(vec![-128, -1])
    );
}

/// Values an `f32` working type rounds: 2^24 + 1 needs 25 significand bits,
/// 2^53 + 1 needs 54, and 1 + 2^-52 needs all 53 of an `f64`.
#[test]
fn decode_keeps_values_a_narrower_float_would_round() {
    let order = ByteOrder::LittleEndian;
    let i32_probe = (1_i32 << 24) + 1;
    let u32_probe = (1_u32 << 24) + 1;
    let i64_probe = -((1_i64 << 53) + 1);
    let u64_probe = u64::MAX;
    let f64_probe = 1.0 + f64::EPSILON;

    let read = |bytes: Vec<u8>, sample_type| {
        SampleBuffer::decode(&bytes, sample_type, order).expect("one sample")
    };
    assert_eq!(
        read(encode(&[i32_probe], order), SampleType::I32)
            .into_vec::<i32>()
            .expect("the stored type"),
        [i32_probe]
    );
    assert_eq!(
        read(encode(&[u32_probe], order), SampleType::U32)
            .into_vec::<u32>()
            .expect("the stored type"),
        [u32_probe]
    );
    assert_eq!(
        read(encode(&[i64_probe], order), SampleType::I64)
            .into_vec::<i64>()
            .expect("the stored type"),
        [i64_probe]
    );
    assert_eq!(
        read(encode(&[u64_probe], order), SampleType::U64)
            .into_vec::<u64>()
            .expect("the stored type"),
        [u64_probe]
    );
    assert_eq!(
        read(encode(&[f64_probe], order), SampleType::F64)
            .into_vec::<f64>()
            .expect("the stored type"),
        [f64_probe]
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
        assert!(matches!(
            err,
            SampleError::PartialSample {
                sample_type: found_type,
                byte_len
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
fn sample_type_descriptors_match_the_primitives() {
    assert_eq!(
        SampleType::ALL.map(SampleType::byte_width),
        [1, 1, 2, 2, 4, 4, 8, 8, 4, 8]
    );
    assert_eq!(
        SampleType::ALL.map(SampleType::name),
        [
            "u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "f32", "f64"
        ]
    );
    let floats: Vec<SampleType> = SampleType::ALL
        .into_iter()
        .filter(|t| t.is_float())
        .collect();
    assert_eq!(floats, [SampleType::F32, SampleType::F64]);
    assert_eq!(SampleType::I64.to_string(), "i64");
}
