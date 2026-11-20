//! Bounded raw DEFLATE for PS3.5 Annex A.5.

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use flate2::{Decompress, FlushDecompress, Status};

pub(super) fn dataset(input: &[u8], allowance: usize) -> Result<Bytes> {
    let mut decoder = Decompress::new(false);
    let mut output = Vec::new();
    let mut block = [0u8; 16 * 1024];
    loop {
        let consumed =
            usize::try_from(decoder.total_in()).context("DICOM deflate input offset overflow")?;
        let produced = decoder.total_out();
        let room = allowance
            .saturating_sub(output.len())
            .saturating_add(1)
            .min(block.len());
        let status = decoder
            .decompress(
                &input[consumed..],
                &mut block[..room],
                FlushDecompress::None,
            )
            .context("invalid deflated DICOM dataset")?;
        let wrote = usize::try_from(decoder.total_out() - produced)?;
        if wrote > allowance - output.len() {
            bail!("deflated DICOM dataset exceeds parse budget of {allowance} bytes");
        }
        output.extend_from_slice(&block[..wrote]);
        let next = usize::try_from(decoder.total_in())?;
        if status == Status::StreamEnd {
            if !matches!(&input[next..], [] | [0]) {
                bail!("deflated DICOM dataset has trailing bytes after the stream");
            }
            return Ok(output.into());
        }
        if next == consumed && wrote == 0 {
            bail!("deflated DICOM dataset is truncated");
        }
    }
}
