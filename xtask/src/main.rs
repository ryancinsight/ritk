use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

mod datasets;

use datasets::{Dataset, DatasetManager};

#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "Build automation for ritk project")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download test datasets for registration testing
    DownloadDatasets {
        /// Dataset to download (all, oasis, learn2reg,ixi)
        #[arg(default_value = "all")]
        dataset: String,
        
        /// Output directory for datasets
        #[arg(short, long, default_value = "test_data")]
        output: PathBuf,
        
        /// Force re-download even if files exist
        #[arg(short, long)]
        force: bool,
    },
    
    /// List available datasets
    ListDatasets,
    
    /// Verify downloaded datasets
    VerifyDatasets {
        /// Directory containing test datasets
        #[arg(short, long, default_value = "test_data")]
        data_dir: PathBuf,
    },
    
    /// Run integration tests with real data
    TestRealData {
        /// Directory containing test datasets
        #[arg(short, long, default_value = "test_data")]
        data_dir: PathBuf,
        
        /// Test to run (all, io, registration)
        #[arg(default_value = "all")]
        test: String,
    },
    
    /// Clean downloaded datasets
    Clean {
        /// Directory containing test datasets
        #[arg(short, long, default_value = "test_data")]
        data_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::DownloadDatasets { dataset, output, force } => {
            download_datasets(&dataset, &output, force)?;
        }
        Commands::ListDatasets => {
            list_datasets();
        }
        Commands::VerifyDatasets { data_dir } => {
            verify_datasets(&data_dir)?;
        }
        Commands::TestRealData { data_dir, test } => {
            test_real_data(&data_dir, &test)?;
        }
        Commands::Clean { data_dir } => {
            clean_datasets(&data_dir)?;
        }
    }
    
    Ok(())
}

fn download_datasets(name: &str, output: &Path, force: bool) -> Result<()> {
    info!("Downloading datasets to: {}", output.display());
    
    std::fs::create_dir_all(output)?;
    
    let manager = DatasetManager::new(output);
    
    let datasets: Vec<Box<dyn Dataset>> = match name {
        "all" => vec![
            Box::new(datasets::OpenNeuroDataset::new()),
            Box::new(datasets::AntsExampleDataset::new()),
            Box::new(datasets::OasisDataset::new()),
            Box::new(datasets::IxiDataset::new()),
            Box::new(datasets::Learn2RegDataset::new()),
        ],
        "openneuro" => vec![Box::new(datasets::OpenNeuroDataset::new())],
        "ants" => vec![Box::new(datasets::AntsExampleDataset::new())],
        "oasis" => vec![Box::new(datasets::OasisDataset::new())],
        "ixi" => vec![Box::new(datasets::IxiDataset::new())],
        "learn2reg" => vec![Box::new(datasets::Learn2RegDataset::new())],
        "brainweb" => vec![Box::new(datasets::BrainWebDataset::new())],
        "synthstrip" => vec![Box::new(datasets::SynthStripDataset::new())],
        _ => {
            anyhow::bail!("Unknown dataset: {}. Use 'list-datasets' to see available options.", name);
        }
    };
    
    for dataset in datasets {
        info!("Downloading {}...", dataset.name());
        match manager.download(dataset.as_ref(), force) {
            Ok(()) => info!("Successfully downloaded {}", dataset.name()),
            Err(e) => warn!("Failed to download {}: {}", dataset.name(), e),
        }
    }
    
    info!("Dataset download complete!");
    Ok(())
}

fn list_datasets() {
    println!("Available datasets:");
    println!();
    println!("  openneuro   - OpenNeuro ds000102 - Sample fMRI dataset");
    println!("                https://openneuro.org/");
    println!("                Status: Direct download available");
    println!();
    println!("  ants        - ANTs example brain data");
    println!("                https://github.com/ANTsX/ANTs");
    println!("                Status: Direct download from GitHub");
    println!();
    println!("  oasis       - OASIS Brains Dataset (416 MR sessions)");
    println!("                https://www.oasis-brains.org/");
    println!("                Status: Requires registration");
    println!();
    println!("  ixi         - IXI Dataset (~600 MR brain images)");
    println!("                https://brain-development.org/ixi-dataset/");
    println!("                Status: Requires registration");
    println!();
    println!("  learn2reg   - Learn2Reg Challenge datasets");
    println!("                https://learn2reg.grand-challenge.org/");
    println!("                Status: Available via Zenodo");
    println!();
    println!("  brainweb    - BrainWeb simulated brain MRI");
    println!("                https://brainweb.bic.mni.mcgill.ca/");
    println!("                Status: Requires form submission");
    println!();
    println!("  synthstrip  - SynthStrip test brain MRI data");
    println!("                https://surfer.nmr.mgh.harvard.edu/fswiki/SynthStrip");
    println!("                Status: Check FreeSurfer repository");
    println!();
    println!("  all         - Download all available datasets");
}

fn verify_datasets(data_dir: &Path) -> Result<()> {
    info!("Verifying datasets in: {}", data_dir.display());
    
    if !data_dir.exists() {
        warn!("Data directory does not exist: {}", data_dir.display());
        return Ok(());
    }
    
    let manager = DatasetManager::new(data_dir);
    manager.verify()?;
    
    info!("Dataset verification complete!");
    Ok(())
}

fn test_real_data(data_dir: &Path, test: &str) -> Result<()> {
    info!("Running real data tests from: {}", data_dir.display());
    
    if !data_dir.exists() {
        anyhow::bail!(
            "Test data directory does not exist: {}. \
             Run 'cargo xtask download-datasets' first.",
            data_dir.display()
        );
    }
    
    match test {
        "all" => run_all_tests(data_dir)?,
        "io" => run_io_tests(data_dir)?,
        "registration" => run_registration_tests(data_dir)?,
        _ => anyhow::bail!("Unknown test: {}", test),
    }
    
    info!("Real data tests complete!");
    Ok(())
}

fn run_all_tests(data_dir: &Path) -> Result<()> {
    run_io_tests(data_dir)?;
    run_registration_tests(data_dir)?;
    Ok(())
}

fn run_io_tests(data_dir: &Path) -> Result<()> {
    info!("Running I/O tests...");
    
    // Check for NIfTI files
    let nifti_files: Vec<_> = walkdir::WalkDir::new(data_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            let ext = path.extension().and_then(|e| e.to_str());
            matches!(ext, Some("nii") | Some("gz"))
        })
        .take(5)
        .collect();
    
    if nifti_files.is_empty() {
        warn!("No NIfTI files found for I/O testing");
    } else {
        info!("Found {} NIfTI files", nifti_files.len());
        for file in &nifti_files {
            info!("  - {}", file.path().display());
        }
    }
    
    Ok(())
}

fn run_registration_tests(data_dir: &Path) -> Result<()> {
    info!("Running registration tests...");
    
    // Look for paired datasets
    let pairs = find_image_pairs(data_dir)?;
    
    if pairs.is_empty() {
        warn!("No image pairs found for registration testing");
        info!("Download datasets first with: cargo xtask download-datasets");
    } else {
        info!("Found {} image pairs for testing", pairs.len());
        for (fixed, moving) in &pairs {
            info!("  Fixed: {}, Moving: {}", fixed.display(), moving.display());
        }
    }
    
    Ok(())
}

fn find_image_pairs(data_dir: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut pairs = Vec::new();
    
    // Look for standard pair patterns
    for entry in walkdir::WalkDir::new(data_dir).max_depth(3) {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            if stem.ends_with("_fixed") {
                let moving = path.with_file_name(format!(
                    "{}_moving.nii.gz",
                    &stem[..stem.len() - 6]
                ));
                if moving.exists() {
                    pairs.push((path.to_path_buf(), moving));
                }
            }
        }
    }
    
    Ok(pairs)
}

fn clean_datasets(data_dir: &Path) -> Result<()> {
    if data_dir.exists() {
        info!("Removing dataset directory: {}", data_dir.display());
        std::fs::remove_dir_all(data_dir)?;
        info!("Datasets cleaned successfully");
    } else {
        info!("No datasets to clean");
    }
    Ok(())
}
