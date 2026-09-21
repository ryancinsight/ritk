use std::fs::File;
use std::io::{Read, Take};
use std::path::Path;

use anyhow::{bail, Context, Result};
use consus_raster::{jpeg, DecodeLimits, DecodedImage};

// JPEG dimension fields are unsigned 16-bit values. File policy assigns at
// most 256 MiB to encoded input and 100 million encoded-grid pixels. Provider
// working storage is independently capped at eight bytes per allowed grid
// pixel, so component-rich or heavily padded streams may reach that cap first.
const MEBIBYTE: usize = 1024 * 1024;
const MAX_FILE_ENCODED_BYTES: usize = 256 * MEBIBYTE;
const MAX_FILE_PIXELS: usize = 100_000_000;
const WORKING_BYTES_PER_GRID_PIXEL: usize = 8;
const MAX_FILE_WORKING_BYTES: usize = WORKING_BYTES_PER_GRID_PIXEL * MAX_FILE_PIXELS;

pub(crate) fn decode_file(path: &Path) -> Result<DecodedImage> {
    let file = File::open(path)
        .with_context(|| format!("failed to open JPEG file: {}", path.display()))?;
    let reported_len = usize::try_from(
        file.metadata()
            .with_context(|| format!("failed to inspect JPEG file: {}", path.display()))?
            .len(),
    )
    .context("JPEG file length exceeds the host address space")?;
    if reported_len > MAX_FILE_ENCODED_BYTES {
        bail!(
            "JPEG file {} is {} bytes; limit is {} bytes",
            path.display(),
            reported_len,
            MAX_FILE_ENCODED_BYTES
        );
    }

    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(reported_len)
        .context("failed to reserve storage for encoded JPEG file")?;
    let read_bound = u64::try_from(MAX_FILE_ENCODED_BYTES)
        .expect("invariant: JPEG encoded-byte limit fits in u64")
        + 1;
    let mut bounded: Take<File> = file.take(read_bound);
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let count = bounded
            .read(&mut chunk)
            .with_context(|| format!("failed to read JPEG file: {}", path.display()))?;
        if count == 0 {
            break;
        }
        let new_len = bytes
            .len()
            .checked_add(count)
            .context("encoded JPEG length overflow")?;
        if new_len > MAX_FILE_ENCODED_BYTES {
            bail!(
                "JPEG file {} grew beyond the {} byte limit while reading",
                path.display(),
                MAX_FILE_ENCODED_BYTES
            );
        }
        bytes
            .try_reserve(count)
            .context("failed to reserve storage while reading encoded JPEG file")?;
        bytes.extend_from_slice(&chunk[..count]);
    }

    jpeg::decode(
        &bytes,
        DecodeLimits {
            max_encoded_bytes: MAX_FILE_ENCODED_BYTES,
            max_dimension: u32::from(u16::MAX),
            max_pixels: MAX_FILE_PIXELS,
            max_working_bytes: MAX_FILE_WORKING_BYTES,
        },
    )
    .with_context(|| format!("failed to decode JPEG file: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;

    use super::*;
    use tempfile::tempdir;

    #[test]
    fn oversized_file_is_rejected_before_reading() -> Result<()> {
        let directory = tempdir()?;
        let path = directory.path().join("oversized.jpg");
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        file.set_len(
            u64::try_from(MAX_FILE_ENCODED_BYTES)
                .expect("invariant: JPEG encoded-byte limit fits in u64")
                + 1,
        )?;

        let error = decode_file(&path).expect_err("oversized JPEG must be rejected");

        assert!(error.to_string().contains("limit"), "got: {error:#}");
        Ok(())
    }
}
