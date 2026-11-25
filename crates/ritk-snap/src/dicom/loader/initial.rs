//! Default acquisition selection for both workstation hosts.
use crate::LoadedVolume;
use anyhow::{bail, Result};

/// Attempt each ranked acquisition once, in order. A successful index is
/// exactly the number of preceding rejected acquisitions. Keep source paths
/// and DICOM values out of the aggregated failure diagnostic.
pub(crate) fn first_readable(
    count: usize,
    mut load: impl FnMut(usize) -> Result<LoadedVolume>,
) -> Result<(usize, LoadedVolume)> {
    let mut first_error_chain_depth = None;
    for index in 0..count {
        match load(index) {
            Ok(volume) => {
                if index > 0 {
                    tracing::warn!(
                        selected_series_index = index,
                        failed_series_count = index,
                        first_error_chain_depth,
                        "unreadable series skipped while opening study"
                    );
                }
                return Ok((index, volume));
            }
            Err(error) => {
                first_error_chain_depth.get_or_insert_with(|| error.chain().count());
            }
        }
    }
    tracing::warn!(
        series_count = count,
        first_error_chain_depth,
        "no discovered series could be opened"
    );
    bail!("no series in the selected study could be opened")
}
