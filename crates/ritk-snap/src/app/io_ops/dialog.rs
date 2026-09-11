//! File-chooser access and PNG encoding shared by the I/O workflows.
//!
//! `rfd` has no wasm backend, so the shim below stands in for it there and
//! every picker returns `None` -- the workflows compile and refuse rather
//! than the target losing the module.

use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
pub(super) use rfd::FileDialog;

#[cfg(target_arch = "wasm32")]
pub(super) struct FileDialog;

#[cfg(target_arch = "wasm32")]
impl FileDialog {
    pub(super) fn new() -> Self {
        Self
    }

    pub(super) fn set_file_name(self, _name: &str) -> Self {
        self
    }

    pub(super) fn add_filter(self, _name: &str, _extensions: &[&str]) -> Self {
        self
    }

    pub(super) fn pick_file(self) -> Option<std::path::PathBuf> {
        None
    }

    pub(super) fn pick_folder(self) -> Option<std::path::PathBuf> {
        None
    }

    pub(super) fn save_file(self) -> Option<std::path::PathBuf> {
        None
    }
}

pub(super) fn color_image_to_rgb_bytes(color_image: &egui::ColorImage) -> Vec<u8> {
    let mut rgb_bytes = Vec::with_capacity(color_image.pixels.len() * 3);
    for pixel in &color_image.pixels {
        rgb_bytes.extend_from_slice(&[pixel.r(), pixel.g(), pixel.b()]);
    }
    rgb_bytes
}

pub(super) fn save_color_image_png(
    path: &Path,
    color_image: &egui::ColorImage,
) -> anyhow::Result<()> {
    let rgb_bytes = color_image_to_rgb_bytes(color_image);
    let [w, h] = color_image.size;
    image::RgbImage::from_raw(w as u32, h as u32, rgb_bytes)
        .ok_or_else(|| anyhow::anyhow!("buffer length mismatch"))
        .and_then(|img| img.save(path).map_err(anyhow::Error::from))
}
