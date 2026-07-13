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
    kind: Option<String>,
    target: Option<String>,
}

pub(crate) fn verify() -> Result<()> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps", "--locked"])
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
        let manifest_text = fs::read_to_string(&package.manifest_path).with_context(|| {
            format!(
                "failed to read manifest {}",
                package.manifest_path.display()
            )
        })?;
        let manifest: toml::Value = toml::from_str(&manifest_text).with_context(|| {
            format!(
                "failed to parse manifest {}",
                package.manifest_path.display()
            )
        })?;
        for dependency in package
            .dependencies
            .iter()
            .filter(|dependency| dependency.source.is_some() && dependency.path.is_none())
        {
            let manifest_name = dependency.rename.as_deref().unwrap_or(&dependency.name);
            if !inherits_workspace_dependency(
                &manifest,
                manifest_name,
                dependency.kind.as_deref().unwrap_or("normal"),
                dependency.target.as_deref(),
            ) {
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

fn inherits_workspace_dependency(
    manifest: &toml::Value,
    dependency: &str,
    kind: &str,
    target: Option<&str>,
) -> bool {
    let table_name = match kind {
        "normal" => "dependencies",
        "dev" => "dev-dependencies",
        "build" => "build-dependencies",
        _ => return false,
    };
    let scope = target.map_or(Some(manifest), |target| manifest.get("target")?.get(target));
    scope
        .and_then(|scope| scope.get(table_name))
        .and_then(toml::Value::as_table)
        .and_then(|dependencies| dependencies.get(dependency))
        .and_then(toml::Value::as_table)
        .and_then(|declaration| declaration.get("workspace"))
        .and_then(toml::Value::as_bool)
        == Some(true)
}

#[cfg(test)]
mod tests {
    use super::inherits_workspace_dependency;

    #[test]
    fn inheritance_detection_respects_kind_and_target_scope() {
        let manifest = toml::from_str(
            r#"
                [dependencies]
                serde = { workspace = true, features = ["derive"] }

                [dev-dependencies]
                mockito = { workspace = true }

                [target.'cfg(target_arch = "wasm32")'.dependencies]
                getrandom = { workspace = true, features = ["wasm_js"] }
            "#,
        )
        .expect("test manifest must be valid TOML");
        assert!(inherits_workspace_dependency(
            &manifest, "serde", "normal", None
        ));
        assert!(inherits_workspace_dependency(
            &manifest, "mockito", "dev", None
        ));
        assert!(inherits_workspace_dependency(
            &manifest,
            "getrandom",
            "normal",
            Some("cfg(target_arch = \"wasm32\")")
        ));
        assert!(!inherits_workspace_dependency(
            &manifest, "serde", "dev", None
        ));
        assert!(!inherits_workspace_dependency(
            &manifest,
            "getrandom",
            "normal",
            Some("cfg(unix)")
        ));
    }
}
