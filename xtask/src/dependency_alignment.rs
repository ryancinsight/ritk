//! Workspace dependency-inheritance gate.
#![expect(clippy::print_stdout, reason = "ratchet RITK-LINT-1")]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
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

/// Runtime tensor/autograd stacks that Coeus replaces. Coeus is the SSOT for
/// every tensor operation, so the graph must not reach a second one -- and
/// "reach" is the operative word: `onnx-ir` pulled `burn-tensor` in without any
/// manifest in this workspace naming `burn`.
const FORBIDDEN_RUNTIME_STACKS: &[&str] = &["burn", "tch"];

/// Match a crate name against a stack root, so `burn`, `burn-tensor` and
/// `burn-ndarray` all resolve to `burn` while `burner` does not.
fn forbidden_stack(crate_name: &str) -> Option<&'static str> {
    FORBIDDEN_RUNTIME_STACKS.iter().copied().find(|stack| {
        crate_name == *stack
            || crate_name
                .strip_prefix(stack)
                .is_some_and(|rest| rest.starts_with('-'))
    })
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
        let relative = package
            .manifest_path
            .strip_prefix(&metadata.workspace_root)
            .unwrap_or(&package.manifest_path);
        for dependency in &package.dependencies {
            if let Some(stack) = forbidden_stack(&dependency.name) {
                failures.push(format!(
                    "{}: declares the `{stack}` runtime stack ('{}'); tensor operations are Coeus",
                    relative.display(),
                    dependency.name
                ));
            }
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
                failures.push(format!(
                    "{}: dependency '{}' is not inherited from workspace",
                    relative.display(),
                    manifest_name
                ));
            }
        }
    }
    for (stack, crates) in reachable_forbidden_stacks(&metadata.workspace_root)? {
        failures.push(format!(
            "the resolved dependency graph reaches the `{stack}` runtime stack ({}); tensor \
             operations are Coeus",
            crates.into_iter().collect::<Vec<_>>().join(", ")
        ));
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

/// The stacks must be absent from the *resolved* graph, not merely undeclared:
/// `burn` arrived here through `onnx-ir` with no manifest in this workspace
/// naming it.
///
/// Match on the resolved package names rather than `cargo tree -i <stack>`.
/// No package in this graph is called plain `burn` -- the lock carries
/// `burn-tensor`, `burn-backend` and `burn-std` -- so an exact-name inversion
/// reports "did not match any packages" while the stack is present.
/// `cargo tree --format '{p}'` prints one `name version` line per node.
fn reachable_forbidden_stacks(
    workspace_root: &Path,
) -> Result<BTreeMap<&'static str, BTreeSet<String>>> {
    let output = Command::new("cargo")
        .args([
            "tree",
            "--locked",
            "--workspace",
            "--prefix",
            "none",
            "--format",
            "{p}",
        ])
        .current_dir(workspace_root)
        .output()
        .context("failed to run cargo tree")?;
    if !output.status.success() {
        bail!(
            "cargo tree failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let mut found: BTreeMap<&'static str, BTreeSet<String>> = BTreeMap::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Some(name) = line.split_whitespace().next() else {
            continue;
        };
        if let Some(stack) = forbidden_stack(name) {
            found.entry(stack).or_default().insert(name.to_string());
        }
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::{forbidden_stack, inherits_workspace_dependency};

    #[test]
    fn forbidden_stack_matches_the_stack_and_its_crates_but_not_lookalikes() {
        assert_eq!(forbidden_stack("burn"), Some("burn"));
        assert_eq!(forbidden_stack("burn-tensor"), Some("burn"));
        assert_eq!(forbidden_stack("burn-ndarray"), Some("burn"));
        assert_eq!(forbidden_stack("tch"), Some("tch"));
        // A lookalike is not the stack.
        assert_eq!(forbidden_stack("burner"), None);
        // Coeus is the replacement, and `onnx-ir` is not itself a runtime
        // stack -- it is caught by the reachability check instead.
        assert_eq!(forbidden_stack("coeus-tensor"), None);
        assert_eq!(forbidden_stack("onnx-ir"), None);
    }

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
