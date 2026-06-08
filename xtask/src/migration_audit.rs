use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const BURN_TOKENS: &[&str] = &[
    "burn::",
    "burn_ndarray",
    "Tensor<",
    "Tensor::<",
    "TensorData",
    "Shape::new",
    "Autodiff",
    "AutodiffBackend",
    "GradientsParams",
    "Conv3d",
    "Param<",
];

const COEUS_REQUIREMENTS: &[&str] = &[
    "Image tensor construction from shape + f32 slices without Burn TensorData",
    "Rank-generic tensor views: reshape, slice, permute, broadcast, transpose",
    "Host extraction for format writers and CLI boundaries with explicit sync points",
    "Elementwise arithmetic, reductions, matmul, interpolation, and histogram kernels",
    "Reverse-mode autodiff for registration transforms and MI/MSE/NCC/CR metrics",
    "3-D convolution, pooling, activation, parameter, optimizer, and model-module APIs",
    "WGPU backend parity for tensor ops used by RITK registration and model crates",
    "Custom GPU kernels for sparse Parzen histogram accumulation or an equivalent scatter path",
    "CPU reference backend and GPU differential test harness with bounded epsilon",
    "PyO3-compatible conversion layer that keeps Python as a thin binding surface",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BurnMigrationReport {
    pub(crate) manifest_dependencies: Vec<PathBuf>,
    pub(crate) source_references: Vec<SourceReference>,
    pub(crate) by_crate: BTreeMap<String, CrateBurnSurface>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SourceReference {
    pub(crate) path: PathBuf,
    pub(crate) count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CrateBurnSurface {
    pub(crate) manifest_dependency: bool,
    pub(crate) source_reference_count: usize,
}

pub(crate) fn print_burn_migration_audit(root: &Path) -> Result<()> {
    let report = scan_burn_migration_surface(root)?;

    println!("Burn migration audit");
    println!("====================");
    println!();
    println!(
        "Manifest dependencies: {}",
        report.manifest_dependencies.len()
    );
    for path in &report.manifest_dependencies {
        println!("  - {}", path.display());
    }

    println!();
    println!(
        "Source files with Burn-surface tokens: {}",
        report.source_references.len()
    );
    for source in &report.source_references {
        println!("  - {} ({})", source.path.display(), source.count);
    }

    println!();
    println!("Crate summary:");
    for (name, surface) in &report.by_crate {
        println!(
            "  - {name}: manifest={}, source_tokens={}",
            surface.manifest_dependency, surface.source_reference_count
        );
    }

    println!();
    println!("Coeus migration requirements:");
    for requirement in COEUS_REQUIREMENTS {
        println!("  - {requirement}");
    }

    Ok(())
}

pub(crate) fn scan_burn_migration_surface(root: &Path) -> Result<BurnMigrationReport> {
    let mut manifest_dependencies = Vec::new();
    let mut source_references = Vec::new();
    let mut by_crate = BTreeMap::<String, CrateBurnSurface>::new();

    visit_files(root, &mut |path| {
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            return Ok(());
        };

        if file_name == "Cargo.toml" {
            let text = fs::read_to_string(path)
                .with_context(|| format!("failed reading {}", path.display()))?;
            if text.contains("burn =") || text.contains("burn-ndarray") {
                let rel = relative(root, path);
                if let Some(crate_name) = crate_name_from_manifest(root, path) {
                    by_crate.entry(crate_name).or_default().manifest_dependency = true;
                }
                manifest_dependencies.push(rel);
            }
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            let text = fs::read_to_string(path)
                .with_context(|| format!("failed reading {}", path.display()))?;
            let count = BURN_TOKENS
                .iter()
                .map(|token| text.matches(token).count())
                .sum::<usize>();
            if count > 0 {
                if let Some(crate_name) = crate_name_from_source(root, path) {
                    by_crate
                        .entry(crate_name)
                        .or_default()
                        .source_reference_count += count;
                }
                source_references.push(SourceReference {
                    path: relative(root, path),
                    count,
                });
            }
        }

        Ok(())
    })?;

    manifest_dependencies.sort();
    source_references.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(BurnMigrationReport {
        manifest_dependencies,
        source_references,
        by_crate,
    })
}

fn visit_files(root: &Path, visit: &mut impl FnMut(&Path) -> Result<()>) -> Result<()> {
    if should_skip(root) {
        return Ok(());
    }

    for entry in fs::read_dir(root).with_context(|| format!("failed reading {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if should_skip(&path) {
            continue;
        }
        if path.is_dir() {
            visit_files(&path, visit)?;
        } else {
            visit(&path)?;
        }
    }

    Ok(())
}

fn should_skip(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(".git" | "target" | "xtask")
    )
}

fn relative(root: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(root).unwrap_or(path).to_path_buf()
}

fn crate_name_from_manifest(root: &Path, manifest: &Path) -> Option<String> {
    let rel = manifest.strip_prefix(root).ok()?;
    let mut parts = rel.components();
    match parts.next()?.as_os_str().to_str()? {
        "crates" => parts.next()?.as_os_str().to_str().map(str::to_owned),
        "xtask" => Some("xtask".to_owned()),
        _ => None,
    }
}

fn crate_name_from_source(root: &Path, source: &Path) -> Option<String> {
    let rel = source.strip_prefix(root).ok()?;
    let mut parts = rel.components();
    match parts.next()?.as_os_str().to_str()? {
        "crates" => parts.next()?.as_os_str().to_str().map(str::to_owned),
        "xtask" => Some("xtask".to_owned()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn audit_detects_manifest_and_source_burn_surface() {
        let root = temp_root();
        fs::create_dir_all(root.join("crates/ritk-core/src")).unwrap();
        fs::write(
            root.join("crates/ritk-core/Cargo.toml"),
            "[dependencies]\nburn = { workspace = true }\nburn-ndarray = { workspace = true }\n",
        )
        .unwrap();
        fs::write(
            root.join("crates/ritk-core/src/lib.rs"),
            "use burn::tensor::{Shape, Tensor, TensorData};\n\
             fn build<B>() { let _ = Shape::new([1]); let _: Option<Tensor<B, 1>> = None; }\n",
        )
        .unwrap();

        let report = scan_burn_migration_surface(&root).unwrap();

        assert_eq!(
            report.manifest_dependencies,
            vec![PathBuf::from("crates/ritk-core/Cargo.toml")]
        );
        assert_eq!(report.source_references.len(), 1);
        assert_eq!(
            report.by_crate.get("ritk-core"),
            Some(&CrateBurnSurface {
                manifest_dependency: true,
                source_reference_count: 4,
            })
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn audit_ignores_target_directory() {
        let root = temp_root();
        fs::create_dir_all(root.join("target/generated")).unwrap();
        fs::write(
            root.join("target/generated/lib.rs"),
            "use burn::tensor::Tensor;",
        )
        .unwrap();

        let report = scan_burn_migration_surface(&root).unwrap();

        assert!(report.source_references.is_empty());
        assert!(report.manifest_dependencies.is_empty());

        fs::remove_dir_all(root).unwrap();
    }

    fn temp_root() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("ritk-migration-audit-{nanos}"))
    }
}
