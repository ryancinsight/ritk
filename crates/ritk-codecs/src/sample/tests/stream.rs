use std::io::{self, Cursor, Read};

use consus_core::ByteOrder;

use super::{encode, probes};
use crate::sample::{Sample, SampleBuffer};

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
