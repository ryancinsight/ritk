use consus_core::ByteOrder;

use super::{encode, probes};
use crate::sample::{Sample, SampleBuffer, write_samples};

/// Probes padded past one encode block, so the block boundary is crossed.
fn long_probes<T: Sample>() -> Vec<T> {
    probes::<T>().into_iter().cycle().take(20_003).collect()
}

fn write_matches_the_scalar_encoder<T: Sample + PartialEq + std::fmt::Debug>() {
    let values = long_probes::<T>();
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        let mut bytes = Vec::new();
        write_samples(&values, order, &mut bytes).expect("a vector accepts every byte");
        assert_eq!(bytes, encode(&values, order), "{} in {order:?}", T::TYPE);
        let decoded = SampleBuffer::decode(&bytes, T::TYPE, order).expect("whole samples");
        assert_eq!(decoded.into_vec::<T>().expect("the stored type"), values);
    }
}

#[test]
fn write_samples_matches_the_scalar_encoder_for_every_sample_type() {
    write_matches_the_scalar_encoder::<u8>();
    write_matches_the_scalar_encoder::<i8>();
    write_matches_the_scalar_encoder::<u16>();
    write_matches_the_scalar_encoder::<i16>();
    write_matches_the_scalar_encoder::<u32>();
    write_matches_the_scalar_encoder::<i32>();
    write_matches_the_scalar_encoder::<u64>();
    write_matches_the_scalar_encoder::<i64>();
    write_matches_the_scalar_encoder::<f32>();
    write_matches_the_scalar_encoder::<f64>();
}

#[test]
fn write_samples_of_nothing_writes_nothing() {
    let mut bytes = Vec::new();
    write_samples::<f64, _>(&[], ByteOrder::BigEndian, &mut bytes).expect("no bytes to write");
    assert!(bytes.is_empty());
}

#[test]
fn write_samples_returns_the_writer_error() {
    let mut storage = [0_u8; 3];
    let mut sink: &mut [u8] = &mut storage;
    let err = write_samples(&[1_u32], ByteOrder::LittleEndian, &mut sink)
        .expect_err("four bytes do not fit in three");
    assert_eq!(err.kind(), std::io::ErrorKind::WriteZero);
}
