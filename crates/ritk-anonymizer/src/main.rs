//! Local-only DICOM metadata editor. Source studies are never modified.
#[cfg(windows)]
mod windows;

#[cfg(windows)]
fn main() -> anyhow::Result<()> {
    windows::run()
}

#[cfg(not(windows))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("The Métis editor requires Windows and the WebView2 runtime")
}
