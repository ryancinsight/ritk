//! Bounded gzip framing for scalar and multi-frame compute reads.

use crate::MghError;
use flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io::{self, BufReader, Read, Take};

type Reader = BufReader<Take<MultiGzDecoder<BufReader<File>>>>;

/// The decoder consumes at most `limit` expanded bytes. After the image has
/// decoded, drain any footer within that allowance and probe one byte beyond
/// it. EOF proves both an inclusive size bound and a valid final trailer.
pub(super) fn read<T>(
    file: File,
    limit: u64,
    decode: impl FnOnce(&mut Reader) -> Result<T, MghError>,
) -> Result<T, MghError> {
    let gzip = MultiGzDecoder::new(BufReader::new(file));
    let mut reader = BufReader::new(gzip.take(limit));
    let value = decode(&mut reader)?;
    io::copy(&mut reader, &mut io::sink()).map_err(|source| MghError::GzipStream { source })?;
    let mut gzip = reader.into_inner().into_inner();
    if gzip
        .read(&mut [0])
        .map_err(|source| MghError::GzipStream { source })?
        != 0
    {
        return Err(MghError::ExpandedLimit { maximum: limit });
    }
    Ok(value)
}
