//! The sample types MINC2 stores, as HDF5 datatypes.

use anyhow::{bail, Result};
use consus_core::{ByteOrder, Datatype};
use ritk_codecs::sample::SampleType;

/// How an HDF5 datatype stores one voxel: its sample type and byte order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StoredType {
    pub(crate) sample_type: SampleType,
    pub(crate) byte_order: ByteOrder,
}

/// The sample type and byte order an HDF5 datatype stores.
///
/// # Errors
///
/// Returns an error for a datatype that is not an 8-, 16-, 32-, or 64-bit
/// integer, or a 32- or 64-bit float.
pub(crate) fn stored_type(datatype: &Datatype) -> Result<StoredType> {
    let (sample_type, byte_order) = match *datatype {
        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => {
            let sample_type = match (bits.get(), signed) {
                (8, false) => SampleType::U8,
                (8, true) => SampleType::I8,
                (16, false) => SampleType::U16,
                (16, true) => SampleType::I16,
                (32, false) => SampleType::U32,
                (32, true) => SampleType::I32,
                (64, false) => SampleType::U64,
                (64, true) => SampleType::I64,
                (bits, signed) => {
                    bail!("Unsupported MINC2 integer datatype: {bits} bits, signed={signed}")
                }
            };
            (sample_type, byte_order)
        }
        Datatype::Float { bits, byte_order } => {
            let sample_type = match bits.get() {
                32 => SampleType::F32,
                64 => SampleType::F64,
                bits => bail!("Unsupported MINC2 float datatype: {bits} bits"),
            };
            (sample_type, byte_order)
        }
        ref other => bail!("Unsupported MINC2 voxel datatype: {other:?}"),
    };
    Ok(StoredType {
        sample_type,
        byte_order,
    })
}

/// Check that the writer can store `sample_type`.
///
/// MINC2's voxel types are the 8-, 16-, and 32-bit signed and unsigned
/// integers, `f32`, and `f64` (`mitype_t` in libminc's
/// `libsrc2/minc2_structs.h`); it has no 64-bit integer type, so a file with
/// one is unreadable by other MINC software. The reader still accepts 64-bit
/// integer datasets.
///
/// # Errors
///
/// Returns an error naming `sample_type` when MINC2 cannot store it.
pub(crate) fn check_storable(sample_type: SampleType) -> Result<()> {
    if matches!(sample_type, SampleType::U64 | SampleType::I64) {
        bail!(
            "MINC2 cannot store {sample_type} samples; it stores u8, i8, u16, i16, u32, i32, f32, and f64 — convert the image to one of them first"
        );
    }
    Ok(())
}
