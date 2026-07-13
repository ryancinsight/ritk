//! Workspace dependency-inheritance gate.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Metadata {
    workspace_root: PathBuf,
    packages: Vec<Package>,
}

#[derive(Deserialize)]
struct Package {
    manifest_path: PathBuf,
    dependencies: Vec<Dependency>,
}

#[derive(Deserialize)]
struct Dependency {
    name: String,
    rename: Option<String>,
    source: Option<String>,
    path: Option<PathBuf>,
}

pub(crate) fn verify() -> Result<()> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .context("failed to run cargo metadata")?;
    if !output.status.success() {
        bail!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let metadata: Metadata =
        serde_json::from_slice(&output.stdout).context("cargo metadata emitted invalid JSON")?;
    let mut failures = Vec::new();
    for package in metadata.packages {
        let Some(parent) = package.manifest_path.parent() else {
            continue;
        };
        if parent == metadata.workspace_root {
            continue;
        }
        let manifest = fs::read_to_string(&package.manifest_path).with_context(|| {
            format!(
                "failed to read manifest {}",
                package.manifest_path.display()
            )
        })?;
        for dependency in package
            .dependencies
            .iter()
            .filter(|dependency| dependency.source.is_some() && dependency.path.is_none())
        {
            let manifest_name = dependency.rename.as_deref().unwrap_or(&dependency.name);
            if !inherits_workspace_dependency(&manifest, manifest_name) {
                let relative = package
                    .manifest_path
                    .strip_prefix(&metadata.workspace_root)
                    .unwrap_or(&package.manifest_path);
                failures.push(format!(
                    "{}: dependency '{}' is not inherited from workspace",
                    relative.display(),
                    manifest_name
                ));
            }
        }
    }
    if failures.is_empty() {
        println!("Workspace dependency alignment check passed.");
        return Ok(());
    }
    bail!(
        "Workspace dependency alignment check failed:\n{}",
        failures
            .iter()
            .map(|failure| format!(" - {failure}"))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

fn inherits_workspace_dependency(manifest: &str, dependency: &str) -> bool {
    manifest.contains(&format!("{dependency} = {{ workspace = true"))
        || manifest.contains(&format!("{dependency}={{ workspace = true"))
}

#[cfg(test)]
mod tests {
    use super::inherits_workspace_dependency;

    #[test]
    fn inheritance_detection_accepts_both_supported_spacing_forms() {
        assert!(inherits_workspace_dependency(
            "serde = { workspace = true, features = [\"derive\"] }",
            "serde"
        ));
        assert!(inherits_workspace_dependency(
            "serde={ workspace = true }",
            "serde"
        ));
        assert!(!inherits_workspace_dependency(
            "serde = { version = \"1\" }",
            "serde"
        ));
    }
}
