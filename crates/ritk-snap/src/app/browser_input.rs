//! Browser-host file handoff into the RITK viewer shell.
//!
//! Métis owns browser file handles and bounded byte transfer. This adapter
//! only converts the completed, pathless payloads into egui's input carrier so
//! the RITK drop-routing policy can classify and decode them.

/// Take completed Métis browser payloads as egui-compatible files.
pub(crate) fn take_dropped_files() -> Vec<egui::DroppedFile> {
    let Some(batch) = metis_web::take_file_drop() else {
        return Vec::new();
    };

    batch
        .into_files()
        .into_vec()
        .into_iter()
        .map(|payload| {
            let (name, mime, bytes) = payload.into_parts();
            egui::DroppedFile {
                name,
                mime,
                bytes: Some(std::sync::Arc::from(bytes)),
                ..Default::default()
            }
        })
        .collect()
}

/// Append completed Métis browser payloads to the egui input queue.
pub(crate) fn extend_dropped_files(dropped: &mut Vec<egui::DroppedFile>) {
    dropped.extend(take_dropped_files());
}
