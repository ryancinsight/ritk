//! Typed reads of contiguous HDF5 dataset payloads.

use crate::datatype::StoredType;
use anyhow::{Context, Result};
use consus_hdf5::file::Hdf5File;
use ritk_codecs::sample::SampleBuffer;

/// Read `count` samples of `stored` from the contiguous payload at
/// `data_address`, in bounded steps, so a shape taken from untrusted metadata
/// reserves no more than the file backs.
///
/// # Errors
///
/// Returns an error when the payload is shorter than `count` samples or cannot
/// be read; the consus error stays the inner error of the I/O error in the
/// chain.
pub(crate) fn read_payload(
    file: &Hdf5File<std::fs::File>,
    data_address: u64,
    stored: StoredType,
    count: usize,
) -> Result<SampleBuffer> {
    let bytes = count
        .checked_mul(stored.sample_type.byte_width())
        .context("MINC2 voxel data size overflows usize")?;
    let len = u64::try_from(bytes).context("MINC2 voxel data size exceeds u64")?;
    let mut payload = file.contiguous_dataset_reader(data_address, len);
    SampleBuffer::read_from(&mut payload, stored.sample_type, stored.byte_order, count)
        .map_err(anyhow::Error::new)
        .context("Failed to read MINC2 voxel data")
}
