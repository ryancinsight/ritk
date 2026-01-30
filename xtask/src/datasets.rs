use anyhow::Result;
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// Trait for downloadable datasets
pub trait Dataset {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn urls(&self) -> Vec<(&'static str, &'static str)>; // (url, expected_sha256)
    fn extract_data(&self, data: &[u8], dest: &Path) -> Result<()>;
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
            info!("Dataset {} already exists at {}. Use --force to re-download.", 
                   dataset.name(), dataset_dir.display());
            return Ok(());
        }
        
        std::fs::create_dir_all(&dataset_dir)?;
        
        let urls = dataset.urls();
        if urls.is_empty() {
            warn!("No download URLs available for {}", dataset.name());
            info!("Dataset {} requires manual download. See: {}", 
                   dataset.name(), dataset.description());
            return Ok(());
        }
        
        for (url, expected_hash) in urls {
            info!("Downloading from: {}", url);
            
            let filename = url.split('/').last().unwrap_or("download");
            let download_path = dataset_dir.join(filename);
            
            // Download with progress bar
            let data = download_with_progress(url)?;
            
            // Verify hash if provided
            if !expected_hash.is_empty() {
                let actual_hash = hex::encode(Sha256::digest(&data));
                if actual_hash != expected_hash {
                    anyhow::bail!("Hash mismatch for {}. Expected: {}, Got: {}", 
                                  filename, expected_hash, actual_hash);
                }
                info!("Hash verified for {}", filename);
            }
            
            // Extract based on file type
            if filename.ends_with(".tar.gz") || filename.ends_with(".tgz") {
                extract_tar_gz(&data, &dataset_dir)?;
            } else if filename.ends_with(".zip") {
                extract_zip(&data, &dataset_dir)?;
            } else if filename.ends_with(".nii.gz") || filename.ends_with(".nii") {
                // Direct NIfTI file - save as-is
                std::fs::write(&download_path, &data)?;
            } else {
                // Unknown type, save as-is
                std::fs::write(&download_path, &data)?;
            }
        }
        
        info!("Dataset {} downloaded to {}", dataset.name(), dataset_dir.display());
        Ok(())
    }
    
    pub fn verify(&self) -> Result<()> {
        info!("Verifying datasets in {}", self.data_dir.display());
        
        for entry in std::fs::read_dir(&self.data_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                info!("Checking dataset: {}", name);
                
                // Count NIfTI files
                let nifti_count = count_nifti_files(&path)?;
                info!("  Found {} NIfTI files", nifti_count);
            }
        }
        
        Ok(())
    }
}

/// Download data with progress bar
fn download_with_progress(url: &str) -> Result<Vec<u8>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    
    let response = client.get(url).send()
        .map_err(|e| anyhow::anyhow!("Failed to download from {}: {}", url, e))?;
    
    let total_size = response
        .content_length()
        .unwrap_or(0);
    
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
fn extract_tar_gz(data: &[u8], dest: &Path) -> Result<()> {
    let tar = GzDecoder::new(data);
    let mut archive = tar::Archive::new(tar);
    archive.unpack(dest)?;
    Ok(())
}

/// Extract zip archive
fn extract_zip(data: &[u8], dest: &Path) -> Result<()> {
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
fn count_nifti_files(dir: &Path) -> Result<usize> {
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

// ============================================================================
// Dataset Implementations
// ============================================================================

/// Example brain MRI data from OpenNeuro (ds000102 - Flanker task)
/// Small subset for testing
pub struct OpenNeuroDataset;

impl OpenNeuroDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for OpenNeuroDataset {
    fn name(&self) -> &'static str {
        "openneuro"
    }
    
    fn description(&self) -> &'static str {
        "OpenNeuro ds000102 - Sample fMRI dataset for testing"
    }
    
    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // OpenNeuro has S3 buckets with public access
        // Using a small test NIfTI file
        vec![
            ("https://s3.amazonaws.com/openneuro.org/ds000102/sub-01/anat/sub-01_T1w.nii.gz", ""),
        ]
    }
    
    fn extract_data(&self, _data: &[u8], _dest: &Path) -> Result<()> {
        // Files are already .nii.gz, no extraction needed
        Ok(())
    }
}

/// ANTs example data (small test brain)
pub struct AntsExampleDataset;

impl AntsExampleDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for AntsExampleDataset {
    fn name(&self) -> &'static str {
        "ants_example"
    }
    
    fn description(&self) -> &'static str {
        "ANTs example brain data for registration testing"
    }
    
    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // ANTs provides example data on GitHub
        vec![
            ("https://github.com/ANTsX/ANTs/raw/master/Data/Template/S_template.nii.gz", ""),
            ("https://github.com/ANTsX/ANTs/raw/master/Data/Template/S_templateCerebellum.nii.gz", ""),
        ]
    }
    
    fn extract_data(&self, _data: &[u8], _dest: &Path) -> Result<()> {
        Ok(())
    }
}

/// BrainWeb simulated brain data
pub struct BrainWebDataset;

impl BrainWebDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for BrainWebDataset {
    fn name(&self) -> &'static str {
        "brainweb"
    }
    
    fn description(&self) -> &'static str {
        "BrainWeb simulated brain MRI (requires manual download from brainweb.bic.mni.mcgill.ca)"
    }
    
    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // BrainWeb requires form submission
        vec![]
    }
    
    fn extract_data(&self, _data: &[u8], _dest: &Path) -> Result<()> {
        Ok(())
    }
}

/// OASIS Brains Dataset (requires registration)
pub struct OasisDataset;

impl OasisDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for OasisDataset {
    fn name(&self) -> &'static str {
        "oasis"
    }
    
    fn description(&self) -> &'static str {
        "OASIS Brains Dataset (416 MR sessions) - Requires registration at https://www.oasis-brains.org/"
    }
    
    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // OASIS requires registration
        vec![]
    }
    
    fn extract_data(&self, _data: &[u8], _dest: &Path) -> Result<()> {
        Ok(())
    }
}

/// IXI Dataset (small subset)
pub struct IxiDataset;

impl IxiDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for IxiDataset {
    fn name(&self) -> &'static str {
        "ixi"
    }
    
    fn description(&self) -> &'static str {
        "IXI Dataset (~600 MR brain images) - Download from https://brain-development.org/ixi-dataset/"
    }
    
    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // IXI has direct download links for sample data
        vec![]
    }
    
    fn extract_data(&self, _data: &[u8], _dest: &Path) -> Result<()> {
        Ok(())
    }
}

/// Learn2Reg Challenge Dataset
pub struct Learn2RegDataset;

impl Learn2RegDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for Learn2RegDataset {
    fn name(&self) -> &'static str {
        "learn2reg"
    }
    
    fn description(&self) -> &'static str {
        "Learn2Reg Challenge datasets - Available at https://learn2reg.grand-challenge.org/"
    }
    
    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // Learn2Reg datasets are hosted on Zenodo
        vec![]
    }
    
    fn extract_data(&self, _data: &[u8], _dest: &Path) -> Result<()> {
        Ok(())
    }
}

/// SynthStrip brain MRI data (freely available)
pub struct SynthStripDataset;

impl SynthStripDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for SynthStripDataset {
    fn name(&self) -> &'static str {
        "synthstrip"
    }
    
    fn description(&self) -> &'static str {
        "SynthStrip test brain MRI data - FreeSurfer project"
    }
    
    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // SynthStrip has some test data available
        vec![]
    }
    
    fn extract_data(&self, _data: &[u8], _dest: &Path) -> Result<()> {
        Ok(())
    }
}

/// Generate synthetic test data
pub struct SyntheticDataset;

impl SyntheticDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for SyntheticDataset {
    fn name(&self) -> &'static str {
        "synthetic"
    }
    
    fn description(&self) -> &'static str {
        "Synthetic test data generated locally"
    }
    
    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        vec![]
    }
    
    fn extract_data(&self, _data: &[u8], _dest: &Path) -> Result<()> {
        Ok(())
    }
}
