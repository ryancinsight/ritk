use anyhow::Result;
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub fn validate_nifti_payload(filename: &str, data: &[u8]) -> Result<()> {
    if is_html_payload(data) {
        anyhow::bail!(
            "{} appears to be HTML, not NIfTI data (possible bad download URL or auth page)",
            filename
        );
    }

    if filename.ends_with(".nii.gz") {
        if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
            anyhow::bail!("{} is not a gzip stream", filename);
        }
        let mut decoder = GzDecoder::new(data);
        let mut header = [0u8; 4];
        decoder
            .read_exact(&mut header)
            .map_err(|e| anyhow::anyhow!("{} has invalid gzip/NIfTI header: {}", filename, e))?;
        if !looks_like_nifti_header(&header) {
            anyhow::bail!("{} does not contain a valid NIfTI header", filename);
        }
    } else if filename.ends_with(".nii") && (data.len() < 4 || !looks_like_nifti_header(&data[..4]))
    {
        anyhow::bail!("{} does not contain a valid NIfTI header", filename);
    }

    Ok(())
}

pub fn is_html_payload(data: &[u8]) -> bool {
    let prefix_len = data.len().min(1024);
    let prefix = &data[..prefix_len];
    let text = String::from_utf8_lossy(prefix).to_ascii_lowercase();
    text.contains("<!doctype html") || text.contains("<html")
}

pub fn looks_like_nifti_header(header4: &[u8]) -> bool {
    if header4.len() < 4 {
        return false;
    }
    let little = i32::from_le_bytes([header4[0], header4[1], header4[2], header4[3]]);
    let big = i32::from_be_bytes([header4[0], header4[1], header4[2], header4[3]]);
    little == 348 || big == 348 || little == 540 || big == 540
}

pub(super) fn list_nifti_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(dir) {
        let entry = entry?;
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str())
            && (ext == "nii" || ext == "gz")
        {
            files.push(path.to_path_buf());
        }
    }
    Ok(files)
}

/// Download data with progress bar
pub(super) fn download_with_progress(url: &str) -> Result<Vec<u8>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    let response = client
        .get(url)
        .send()
        .map_err(|e| anyhow::anyhow!("Failed to download from {}: {}", url, e))?;

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("#>-"),
    );

    let mut data = Vec::new();
    let mut stream = response;
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = stream.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        data.extend_from_slice(&buffer[..bytes_read]);
        pb.inc(bytes_read as u64);
    }

    pb.finish_with_message("Download complete");
    Ok(data)
}

/// Extract tar.gz archive
pub(super) fn extract_tar_gz(data: &[u8], dest: &Path) -> Result<()> {
    let tar = GzDecoder::new(data);
    let mut archive = tar::Archive::new(tar);
    archive.unpack(dest)?;
    Ok(())
}

/// A ZIP member name that would place a file outside the destination.
#[derive(Debug, thiserror::Error)]
#[error("zip member {name:?} is not a relative path inside the destination: {reason}")]
pub struct UnsafeMemberPath {
    /// Rejected member name.
    pub name: String,
    /// Violated constraint.
    pub reason: &'static str,
}

/// The destination-relative path of a member, or why it is unsafe.
///
/// A name is accepted only when every `/`-separated component is a plain
/// file or directory name: no root, `.` or `..`, and no `\` or `:` that
/// Windows would read as a separator or drive.
pub(crate) fn member_path(name: &str) -> std::result::Result<PathBuf, UnsafeMemberPath> {
    let reject = |reason| UnsafeMemberPath {
        name: name.to_owned(),
        reason,
    };
    if name.contains(['\\', ':']) {
        return Err(reject("contains a backslash or colon"));
    }
    if name.starts_with('/') {
        return Err(reject("is absolute"));
    }
    let mut path = PathBuf::new();
    for component in name.split('/').filter(|component| !component.is_empty()) {
        if component == "." || component == ".." {
            return Err(reject("contains a `.` or `..` component"));
        }
        path.push(component);
    }
    if path.as_os_str().is_empty() {
        return Err(reject("is empty"));
    }
    Ok(path)
}

/// Extracts a zip archive into `dest`.
///
/// Every member name is validated before anything is written, so one unsafe
/// name ([`UnsafeMemberPath`]) leaves `dest` untouched. Members are written
/// as new regular files, never over an existing path, and a symbolic link on
/// the way to a member fails the extraction, so no write follows a link out
/// of `dest`.
pub fn extract_zip(data: &[u8], dest: &Path) -> Result<()> {
    let mut archive = consus_zip::ZipReader::new(std::io::Cursor::new(data))?;
    let members = archive
        .entries()
        .iter()
        .map(|entry| Ok((member_path(entry.name())?, entry.name().ends_with('/'))))
        .collect::<std::result::Result<Vec<_>, UnsafeMemberPath>>()?;
    for (index, (relative, is_directory)) in members.into_iter().enumerate() {
        let target = dest.join(&relative);
        let directory = if is_directory {
            relative.as_path()
        } else {
            relative.parent().unwrap_or(Path::new(""))
        };
        create_dir_without_links(dest, directory)?;
        if !is_directory {
            let bytes = archive.read_entry(index)?;
            std::fs::File::create_new(&target)?.write_all(&bytes)?;
        }
    }
    Ok(())
}

/// Creates `dest/relative` component by component, refusing symbolic links.
fn create_dir_without_links(dest: &Path, relative: &Path) -> Result<()> {
    let mut current = dest.to_path_buf();
    for component in relative.components() {
        current.push(component);
        match std::fs::symlink_metadata(&current) {
            Ok(meta) if meta.file_type().is_symlink() => {
                anyhow::bail!("{} is a symbolic link", current.display())
            }
            Ok(meta) if meta.is_dir() => {}
            Ok(_) => anyhow::bail!("{} exists and is not a directory", current.display()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                std::fs::create_dir(&current)?;
            }
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

/// Count NIfTI files in directory
pub(super) fn count_nifti_files(dir: &Path) -> Result<usize> {
    let mut count = 0;

    for entry in walkdir::WalkDir::new(dir) {
        let entry = entry?;
        let path = entry.path();

        if let Some(ext) = path.extension().and_then(|e| e.to_str())
            && (ext == "nii" || ext == "gz")
        {
            count += 1;
        }
    }

    Ok(count)
}
