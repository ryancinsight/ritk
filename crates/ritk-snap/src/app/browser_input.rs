//! Browser-host file handoff into the RITK viewer shell.
//!
//! Métis owns browser file handles and bounded byte transfer. This adapter
//! converts completed pathless payloads into RITK's format-neutral input
//! carrier so the drop-routing policy can classify and decode them.

use crate::ui::DroppedInput;

/// Take completed Métis browser payloads as RITK-owned inputs.
pub(crate) fn take_dropped_files() -> Vec<DroppedInput> {
    let Some(batch) = metis_web::take_file_drop() else {
        return Vec::new();
    };

    batch
        .into_files()
        .into_vec()
        .into_iter()
        .map(|payload| {
            let (name, mime, bytes) = payload.into_parts();
            DroppedInput::new(None, name, mime, Some(std::sync::Arc::from(bytes)))
        })
        .collect()
}
