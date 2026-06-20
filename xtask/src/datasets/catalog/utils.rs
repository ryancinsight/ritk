use anyhow::Result;
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::Read;
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
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if ext == "nii" || ext == "gz" {
                files.push(path.to_path_buf());
            }
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

/// Extract zip archive
pub(super) fn extract_zip(data: &[u8], dest: &Path) -> Result<()> {
    let reader = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(reader)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = dest.join(file.name());

        if file.name().ends_with('/') {
            std::fs::create_dir_all(&outpath)?;
        } else {
            if let Some(parent) = outpath.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut outfile = File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
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

        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if ext == "nii" || ext == "gz" {
                count += 1;
            }
        }
    }

    Ok(count)
}
