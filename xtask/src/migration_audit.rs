use anyhow::bail;
use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const BURN_TOKENS: &[&str] = &[
    "burn::",
    "burn_ndarray",
    "ritk_image::tensor",
    "TensorData",
    "Shape::new",
    "Autodiff",
    "AutodiffBackend",
    "GradientsParams",
    "Conv3d",
    "Param<",
];

// Standalone legacy crates the Coeus/Atlas migration will remove in addition
// to Burn. These track direct (non-Burn-bundled) usage so re-imports after the
// first migration get flagged by CI via the `xtask/burn_surface.allowlist`
// drift detector.
const STANDALONE_LEGACY_DEPS: &[&str] = &["approx", "num-traits", "rustfft", "ndarray"];

/// Burn / burn-bundled manifest dep names. The burn-side manifest check
/// routes through the same comment + section-header-aware matcher as the
/// standalone side, so false positives like `# historical burn = "0.17"` or
/// unrelated `burn-cache = "..."` dev-deps do not flip the per-crate flag.
const LEGACY_BURN_DEPS: &[&str] = &["burn", "burn-ndarray"];

const STANDALONE_LEGACY_TOKENS: &[&str] = &["approx::", "num_traits::", "rustfft::", "ndarray::"];

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

const ALLOWLIST_REL_PATH: &str = "xtask/burn_surface.allowlist";

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
    pub(crate) standalone_manifest_dependency: bool,
    pub(crate) standalone_source_reference_count: usize,
}

pub(crate) fn print_burn_migration_audit(root: &Path) -> Result<()> {
    let report = scan_burn_migration_surface(root)?;
    let diff = compare_with_allowlist(root, &report)?;

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
    println!("Crate summary (burn / standalone legacy):");
    for (name, surface) in &report.by_crate {
        println!(
            "  - {name}: burn(dep={}, tokens={}), standalone(dep={}, tokens={})",
            surface.manifest_dependency,
            surface.source_reference_count,
            surface.standalone_manifest_dependency,
            surface.standalone_source_reference_count
        );
    }

    println!();
    println!("Coeus migration requirements:");
    for requirement in COEUS_REQUIREMENTS {
        println!("  - {requirement}");
    }

    println!();
    if diff.new_entries.is_empty() {
        println!("Allowlist status: clean");
    } else {
        println!(
            "Allowlist drift: {} new Burn surfaces not in {}",
            diff.new_entries.len(),
            ALLOWLIST_REL_PATH
        );
        for entry in &diff.new_entries {
            println!("  - {entry}");
        }
    }

    if !diff.stale_entries.is_empty() {
        println!();
        println!("Allowlist cleanup candidates (already migrated):");
        for entry in &diff.stale_entries {
            println!("  - {entry}");
        }
    }

    if !diff.new_entries.is_empty() {
        bail!(
            "legacy migration allowlist drift detected (burn or standalone); run `cargo run -p xtask -- refresh-burn-allowlist` after intentional changes"
        );
    }

    Ok(())
}

pub(crate) fn refresh_burn_allowlist(root: &Path) -> Result<()> {
    let report = scan_burn_migration_surface(root)?;
    let mut entries = BTreeSet::new();

    for path in &report.manifest_dependencies {
        entries.insert(manifest_allowlist_entry(path));
    }
    for source in &report.source_references {
        entries.insert(source_allowlist_entry(&source.path));
    }

    let allowlist_path = root.join(ALLOWLIST_REL_PATH);
    if let Some(parent) = allowlist_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed creating {}", parent.display()))?;
    }

    let mut body = String::from(
        "# Auto-generated by `cargo run -p xtask -- refresh-burn-allowlist`\n\
         # Each entry tracks an approved Burn surface during Coeus migration.\n",
    );
    for entry in entries {
        body.push_str(&entry);
        body.push('\n');
    }

    fs::write(&allowlist_path, body)
        .with_context(|| format!("failed writing {}", allowlist_path.display()))?;
    println!("Updated {}", allowlist_path.display());
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
            // Use the comment + section-header-aware matcher on the burn side
            // too, so a `# historical burn = "0.17"` comment or a
            // `burn-cache = ...` dev-dep does not flip the flag incorrectly.
            let has_burn = has_any_manifest_dep(&text, LEGACY_BURN_DEPS);
            let has_standalone = has_any_manifest_dep(&text, STANDALONE_LEGACY_DEPS);
            if has_burn || has_standalone {
                let rel = relative(root, path);
                if let Some(crate_name) = crate_name_from_manifest(root, path) {
                    let entry = by_crate.entry(crate_name).or_default();
                    if has_burn {
                        entry.manifest_dependency = true;
                    }
                    if has_standalone {
                        entry.standalone_manifest_dependency = true;
                    }
                }
                manifest_dependencies.push(rel);
            }
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            let text = fs::read_to_string(path)
                .with_context(|| format!("failed reading {}", path.display()))?;
            let burn_count = BURN_TOKENS
                .iter()
                .map(|token| text.matches(token).count())
                .sum::<usize>();
            let standalone_count = STANDALONE_LEGACY_TOKENS
                .iter()
                .map(|token| text.matches(token).count())
                .sum::<usize>();
            let count = burn_count + standalone_count;
            if count > 0 {
                if let Some(crate_name) = crate_name_from_source(root, path) {
                    let entry = by_crate.entry(crate_name).or_default();
                    entry.source_reference_count += burn_count;
                    entry.standalone_source_reference_count += standalone_count;
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

fn has_any_manifest_dep(text: &str, deps: &[&str]) -> bool {
    text.lines().any(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('[') {
            return false;
        }
        let Some((raw_key, _)) = trimmed.split_once('=') else {
            return false;
        };
        let key = raw_key.trim();
        deps.iter()
            .any(|dep| key == *dep || key == format!("{dep}.workspace"))
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct AllowlistDiff {
    new_entries: Vec<String>,
    stale_entries: Vec<String>,
}

fn compare_with_allowlist(root: &Path, report: &BurnMigrationReport) -> Result<AllowlistDiff> {
    let allowlist_path = root.join(ALLOWLIST_REL_PATH);
    if !allowlist_path.exists() {
        bail!(
            "missing {}; run `cargo run -p xtask -- refresh-burn-allowlist`",
            allowlist_path.display()
        );
    }

    let allowed = load_allowlist(&allowlist_path)?;
    let mut current = BTreeSet::new();

    for path in &report.manifest_dependencies {
        current.insert(manifest_allowlist_entry(path));
    }
    for source in &report.source_references {
        current.insert(source_allowlist_entry(&source.path));
    }

    let new_entries = current
        .difference(&allowed)
        .cloned()
        .collect::<Vec<String>>();
    let stale_entries = allowed
        .difference(&current)
        .cloned()
        .collect::<Vec<String>>();

    Ok(AllowlistDiff {
        new_entries,
        stale_entries,
    })
}

fn load_allowlist(path: &Path) -> Result<BTreeSet<String>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed reading {}", path.display()))?;
    let mut entries = BTreeSet::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        entries.insert(trimmed.to_owned());
    }
    Ok(entries)
}

fn manifest_allowlist_entry(path: &Path) -> String {
    format!("manifest:{}", normalized(path))
}

fn source_allowlist_entry(path: &Path) -> String {
    format!("source:{}", normalized(path))
}

fn normalized(path: &Path) -> String {
    path.display().to_string().replace('\\', "/")
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
            "use ritk_image::tensor::{Shape, Tensor, TensorData};\n\
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
                source_reference_count: 3,
                ..Default::default()
            })
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn audit_does_not_classify_coeus_tensor_syntax_as_burn() {
        let root = temp_root();
        fs::create_dir_all(root.join("crates/ritk-transform/src")).unwrap();
        fs::write(
            root.join("crates/ritk-transform/Cargo.toml"),
            "[dependencies]\ncoeus-tensor = { workspace = true }\n",
        )
        .unwrap();
        fs::write(
            root.join("crates/ritk-transform/src/lib.rs"),
            "use coeus_tensor::Tensor;\nstruct Field<B>(Tensor<f32, B>);\n",
        )
        .unwrap();

        let report = scan_burn_migration_surface(&root).unwrap();

        assert!(report.source_references.is_empty());
        assert!(report.manifest_dependencies.is_empty());
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

    #[test]
    fn audit_detects_standalone_legacy_surface() {
        let root = temp_root();
        fs::create_dir_all(root.join("crates/ritk-statistics/src/")).unwrap();
        fs::write(
            root.join("crates/ritk-statistics/Cargo.toml"),
            "[dependencies]\nnum-traits = \"0.2\"\napprox = \"0.5\"\n",
        )
        .unwrap();
        fs::write(
            root.join("crates/ritk-statistics/src/lib.rs"),
            "use approx::assert_relative_eq;\n\
             use num_traits::Float;\n\
             use ndarray::{Array1, Array2};\n",
        )
        .unwrap();

        let report = scan_burn_migration_surface(&root).unwrap();

        assert_eq!(
            report.manifest_dependencies,
            vec![PathBuf::from("crates/ritk-statistics/Cargo.toml")]
        );
        assert_eq!(report.source_references.len(), 1);
        assert_eq!(
            report.by_crate.get("ritk-statistics"),
            Some(&CrateBurnSurface {
                manifest_dependency: false,
                source_reference_count: 0,
                standalone_manifest_dependency: true,
                standalone_source_reference_count: 3,
            })
        );

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
