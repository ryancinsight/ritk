//! Sample decoding, streaming, conversion, encoding, and rescale, checked
//! against consus-core's scalar codec and the std cast lattice.

use super::Sample;
use consus_core::{ByteOrder, read_integer, write_integer};

mod conversion;
mod decode;
mod encode;
mod lattice;
mod rescale;
mod stream;

/// Encode `values` in `order` one scalar at a time through consus-core's
/// scalar writer, independent of the bulk decoder under test.
fn encode<T: Sample>(values: &[T], order: ByteOrder) -> Vec<u8> {
    let width = T::TYPE.byte_width();
    let mut bytes = vec![0_u8; values.len() * width];
    for (value, slot) in values.iter().zip(bytes.chunks_exact_mut(width)) {
        write_integer(slot, *value, order).expect("invariant: slot holds exactly one sample");
    }
    bytes
}

/// Zero, one, a value exercising every byte, and minus one represented by `T`.
fn probes<T: Sample>() -> Vec<T> {
    let bytes: Vec<u8> = (1..=T::TYPE.byte_width())
        .map(|b| u8::try_from(b * 0x11).expect("invariant: at most 8 * 0x11 = 0x88"))
        .collect();
    let every_byte =
        read_integer::<T>(&bytes, ByteOrder::LittleEndian).expect("one sample of bytes");
    vec![T::zero(), T::one(), every_byte, T::from_signed_sample(-1)]
}
