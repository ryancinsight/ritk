use anyhow::Result;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

mod impls;
pub(crate) mod utils;

pub use impls::{
    AntsExampleDataset, BrainWebDataset, IxiDataset, Learn2RegDataset, OasisDataset,
    OpenNeuroDataset, SynthStripDataset,
};

/// Trait for downloadable datasets
pub trait Dataset {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn urls(&self) -> Vec<(&'static str, &'static str)>; // (url, expected_sha256)
}

/// Manager for dataset operations
pub struct DatasetManager {
    data_dir: PathBuf,
}

impl DatasetManager {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
        }
    }

    pub fn download(&self, dataset: &dyn Dataset, force: bool) -> Result<()> {
        let dataset_dir = self.data_dir.join(dataset.name());

        if dataset_dir.exists() && !force {
            info!(
                "Dataset {} already exists at {}. Use --force to re-download.",
                dataset.name(),
                dataset_dir.display()
            );
            return Ok(());
        }

        std::fs::create_dir_all(&dataset_dir)?;

        let urls = dataset.urls();
        if urls.is_empty() {
            warn!("No download URLs available for {}", dataset.name());
            info!(
                "Dataset {} requires manual download. See: {}",
                dataset.name(),
                dataset.description()
            );
            return Ok(());
        }

        for (url, expected_hash) in urls {
            info!("Downloading from: {}", url);

            let filename = url.split('/').next_back().unwrap_or("download");
            let download_path = dataset_dir.join(filename);

            // Download with progress bar
            let data = utils::download_with_progress(url)?;

            // Verify hash if provided
            if !expected_hash.is_empty() {
                use sha2::{Digest, Sha256};
                let actual_hash = hex::encode(Sha256::digest(&data));
                if actual_hash != expected_hash {
                    anyhow::bail!(
                        "Hash mismatch for {}. Expected: {}, Got: {}",
                        filename,
                        expected_hash,
                        actual_hash
                    );
                }
                info!("Hash verified for {}", filename);
            }

            // Extract based on file type
            if filename.ends_with(".tar.gz") || filename.ends_with(".tgz") {
                utils::extract_tar_gz(&data, &dataset_dir)?;
            } else if filename.ends_with(".zip") {
                utils::extract_zip(&data, &dataset_dir)?;
            } else if filename.ends_with(".nii.gz") || filename.ends_with(".nii") {
                utils::validate_nifti_payload(filename, &data)?;
                // Direct NIfTI file - save as-is
                std::fs::write(&download_path, &data)?;
            } else {
                // Unknown type, save as-is
                std::fs::write(&download_path, &data)?;
            }
        }

        info!(
            "Dataset {} downloaded to {}",
            dataset.name(),
            dataset_dir.display()
        );
        Ok(())
    }

    pub fn verify(&self) -> Result<()> {
        info!("Verifying datasets in {}", self.data_dir.display());
        let mut invalid_nifti_files = Vec::new();

        for entry in std::fs::read_dir(&self.data_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                info!("Checking dataset: {}", name);

                // Count NIfTI files
                let nifti_count = utils::count_nifti_files(&path)?;
                info!("  Found {} NIfTI files", nifti_count);

                for nifti_path in utils::list_nifti_files(&path)? {
                    let data = std::fs::read(&nifti_path)?;
                    let filename = nifti_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("<unknown>");
                    if let Err(err) = utils::validate_nifti_payload(filename, &data) {
                        invalid_nifti_files.push(format!("{} ({})", nifti_path.display(), err));
                    }
                }
            }
        }

        if !invalid_nifti_files.is_empty() {
            anyhow::bail!(
                "Found invalid NIfTI payload(s):\n{}",
                invalid_nifti_files.join("\n")
            );
        }

        Ok(())
    }
}
