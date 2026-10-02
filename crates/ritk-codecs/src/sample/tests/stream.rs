use std::io::{self, Cursor, Read};

use consus_core::ByteOrder;

use super::{encode, probes};
use crate::sample::{count_payload_bytes, validate_remaining_payload, Sample, SampleBuffer};

/// Probes padded past one 16 KiB stream step of every sample width, so the
/// step boundary is crossed with a partial final step.
fn long_probes<T: Sample>() -> Vec<T> {
    probes::<T>()
        .into_iter()
        .cycle()
        .take(16 * 1024 + 3)
        .collect()
}

/// A reader that yields at most three bytes per call.
struct Trickle<'a>(&'a [u8]);

impl Read for Trickle<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = buf.len().min(3).min(self.0.len());
        buf[..n].copy_from_slice(&self.0[..n]);
        self.0 = &self.0[n..];
        Ok(n)
    }
}

fn read_from_matches_the_scalar_encoder<T: Sample + PartialEq + std::fmt::Debug>() {
    let values = long_probes::<T>();
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        let bytes = encode(&values, order);
        let whole = SampleBuffer::read_from(&mut Cursor::new(&bytes), T::TYPE, order, values.len())
            .expect("every sample is present");
        assert_eq!(whole.into_vec::<T>().expect("the stored type"), values);
        let trickled = SampleBuffer::read_from(&mut Trickle(&bytes), T::TYPE, order, values.len())
            .expect("short reads still fill every sample");
        assert_eq!(trickled.into_vec::<T>().expect("the stored type"), values);
    }
}

#[test]
fn read_from_matches_the_scalar_encoder_for_every_sample_type() {
    read_from_matches_the_scalar_encoder::<u8>();
    read_from_matches_the_scalar_encoder::<i8>();
    read_from_matches_the_scalar_encoder::<u16>();
    read_from_matches_the_scalar_encoder::<i16>();
    read_from_matches_the_scalar_encoder::<u32>();
    read_from_matches_the_scalar_encoder::<i32>();
    read_from_matches_the_scalar_encoder::<u64>();
    read_from_matches_the_scalar_encoder::<i64>();
    read_from_matches_the_scalar_encoder::<f32>();
    read_from_matches_the_scalar_encoder::<f64>();
}

/// A stream one byte short of `count` samples ends at the last sample, which
/// the error names.
#[test]
fn read_from_names_the_first_missing_sample() {
    let values = long_probes::<i32>();
    let mut bytes = encode(&values, ByteOrder::BigEndian);
    bytes.pop();
    let err = SampleBuffer::read_from(
        &mut Cursor::new(&bytes),
        crate::sample::SampleType::I32,
        ByteOrder::BigEndian,
        values.len(),
    )
    .expect_err("one byte is missing");
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    let last = values.len() - 1;
    assert_eq!(
        err.to_string(),
        format!("stream ended at i32 sample {last} of {}", values.len())
    );
}

/// A header can declare far more samples than the stream holds; the read
/// fails on the stream's end rather than reserving the declared count.
#[test]
fn read_from_a_short_stream_never_reserves_the_declared_count() {
    let err = SampleBuffer::read_from(
        &mut Cursor::new([0_u8; 8]),
        crate::sample::SampleType::F64,
        ByteOrder::LittleEndian,
        usize::MAX / 8,
    )
    .expect_err("one sample cannot fill the declared count");
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    assert!(err.to_string().contains("f64 sample 1 of"), "{err}");
}

#[test]
fn payload_count_checks_exact_length_and_stops_after_one_excess_byte() {
    let mut short = Cursor::new(b"abc".as_slice());
    assert_eq!(
        count_payload_bytes(&mut short, 4).expect("read succeeds"),
        Some(3)
    );

    let mut exact = Cursor::new(b"abcd".as_slice());
    assert_eq!(
        count_payload_bytes(&mut exact, 4).expect("read succeeds"),
        Some(4)
    );
    assert_eq!(exact.position(), 4);

    let mut excess = Cursor::new(b"abcdef".as_slice());
    assert_eq!(
        count_payload_bytes(&mut excess, 4).expect("read succeeds"),
        None
    );
    assert_eq!(excess.position(), 5);
}

#[test]
fn remaining_payload_validation_restores_position_and_classifies_length() {
    let mut exact = Cursor::new(b"headerdata".as_slice());
    exact.set_position(6);
    validate_remaining_payload(&mut exact, 4).expect("four bytes remain");
    assert_eq!(exact.position(), 6);

    let mut short = Cursor::new(b"headerdat".as_slice());
    short.set_position(6);
    let error = validate_remaining_payload(&mut short, 4).expect_err("three bytes remain");
    assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    assert_eq!(short.position(), 6);

    let mut excess = Cursor::new(b"headerdata!".as_slice());
    excess.set_position(6);
    let error = validate_remaining_payload(&mut excess, 4).expect_err("five bytes remain");
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert_eq!(excess.position(), 6);
}
