//! Plan a copy of a study folder tree into a separate output folder.
//!
//! Tools that rewrite a study never modify their source. [`ExportTree::plan`]
//! checks that before any file is written: the output folder may not be the
//! source, lie inside it, or contain it, and must be empty. The check holds
//! for the file system as it was when the plan was made; nothing guards
//! against the output folder changing between planning and writing.

use std::path::{Path, PathBuf};

/// Why an export could not be planned.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExportTreeError {
    /// The output folder is the source, inside it, or contains it.
    #[error("output folder {output:?} overlaps the source {input:?}")]
    OutputOverlapsInput {
        /// Resolved source path.
        input: PathBuf,
        /// Resolved output path.
        output: PathBuf,
    },
    /// The output folder already holds entries.
    #[error("output folder {0:?} is not empty")]
    OutputNotEmpty(PathBuf),
    /// A file-system step failed.
    #[error("could not {action} {path:?}")]
    Io {
        /// The step that failed.
        action: &'static str,
        /// The path it was applied to.
        path: PathBuf,
        /// The underlying failure.
        #[source]
        source: std::io::Error,
    },
}

pub(crate) fn io_error(
    action: &'static str,
    path: &Path,
) -> impl FnOnce(std::io::Error) -> ExportTreeError {
    let path = path.to_path_buf();
    move |source| ExportTreeError::Io {
        action,
        path,
        source,
    }
}

/// One source file and its mirrored path below the output folder.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ExportEntry {
    /// Resolved source file.
    pub(crate) source: PathBuf,
    /// The output folder joined with the file's path below the source
    /// folder, or with its own name for a single-file source. A caller that
    /// names its outputs otherwise ignores it.
    pub(crate) target: PathBuf,
}

/// The regular files of a source, the resolved output folder, and each
/// file's mirrored path below it.
#[derive(Debug)]
pub(crate) struct ExportTree {
    source: PathBuf,
    destination: PathBuf,
    entries: Vec<ExportEntry>,
}

impl ExportTree {
    /// Resolve `input` and `output` and list the source files in path order.
    ///
    /// `input` is a folder or a single file. `output` is not created; a
    /// missing output folder must have an existing parent. Symbolic links
    /// below `input` are not followed.
    ///
    /// # Errors
    /// Returns [`ExportTreeError`] when a path cannot be resolved or read,
    /// when the two overlap, or when the output folder holds entries.
    pub(crate) fn plan(input: &Path, output: &Path) -> Result<Self, ExportTreeError> {
        let source = input.canonicalize().map_err(io_error("resolve", input))?;
        let destination = resolve_output(output)?;
        if destination.starts_with(&source) || source.starts_with(&destination) {
            return Err(ExportTreeError::OutputOverlapsInput {
                input: source,
                output: destination,
            });
        }
        if destination
            .try_exists()
            .map_err(io_error("inspect", &destination))?
            && std::fs::read_dir(&destination)
                .map_err(io_error("read", &destination))?
                .next()
                .is_some()
        {
            return Err(ExportTreeError::OutputNotEmpty(destination));
        }
        let outside = |path: &Path| ExportTreeError::Io {
            action: "place",
            path: path.to_path_buf(),
            source: std::io::ErrorKind::InvalidInput.into(),
        };
        let root = if source.is_file() {
            source.parent().ok_or_else(|| outside(&source))?
        } else {
            source.as_path()
        };
        let entries = source_files(&source)?
            .into_iter()
            .map(|file| {
                let below = file.strip_prefix(root).map_err(|_| outside(&file))?;
                let target = destination.join(below);
                Ok(ExportEntry {
                    source: file,
                    target,
                })
            })
            .collect::<Result<Vec<_>, ExportTreeError>>()?;
        Ok(Self {
            source,
            destination,
            entries,
        })
    }

    /// The resolved source path.
    pub(crate) fn source(&self) -> &Path {
        &self.source
    }

    /// The resolved output folder.
    pub(crate) fn destination(&self) -> &Path {
        &self.destination
    }

    /// Source files in path order, each with its mirrored target: its
    /// path below the source folder, or its own name for a single-file source,
    /// joined below the output folder.
    pub(crate) fn entries(&self) -> &[ExportEntry] {
        &self.entries
    }
}

/// Resolve `output` without creating it: an existing folder is canonicalized,
/// a missing one is placed under its canonicalized parent.
fn resolve_output(output: &Path) -> Result<PathBuf, ExportTreeError> {
    if output.try_exists().map_err(io_error("inspect", output))? {
        return output.canonicalize().map_err(io_error("resolve", output));
    }
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = output.file_name().ok_or_else(|| ExportTreeError::Io {
        action: "name",
        path: output.to_path_buf(),
        source: std::io::ErrorKind::InvalidInput.into(),
    })?;
    Ok(parent
        .canonicalize()
        .map_err(io_error("resolve", parent))?
        .join(name))
}

fn source_files(source: &Path) -> Result<Vec<PathBuf>, ExportTreeError> {
    if source.is_file() {
        return Ok(vec![source.to_path_buf()]);
    }
    let mut files = Vec::new();
    let mut pending = vec![source.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory).map_err(io_error("read", &directory))? {
            let entry = entry.map_err(io_error("read", &directory))?;
            let path = entry.path();
            let kind = entry.file_type().map_err(io_error("inspect", &path))?;
            if kind.is_dir() {
                pending.push(path);
            } else if kind.is_file() {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::{ExportEntry, ExportTree, ExportTreeError};
    use std::path::Path;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn tree(root: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(root.join("b/deep"))?;
        std::fs::write(root.join("b/deep/two"), "2")?;
        std::fs::write(root.join("c"), "3")?;
        std::fs::write(root.join("a"), "1")
    }

    #[test]
    fn files_are_listed_in_path_order_and_mapped_below_the_output() -> TestResult {
        let source = tempfile::tempdir()?;
        let parent = tempfile::tempdir()?;
        tree(source.path())?;
        let output = parent.path().join("export");

        let plan = ExportTree::plan(source.path(), &output)?;

        let root = source.path().canonicalize()?;
        let destination = parent.path().canonicalize()?.join("export");
        assert_eq!(plan.source(), root);
        assert_eq!(plan.destination(), destination);
        // A directory walk yields `a`, `c`, then `b/deep/two`; path order
        // places the nested file between them.
        let expected = ["a", "b/deep/two", "c"].map(|file| ExportEntry {
            source: root.join(file),
            target: destination.join(file),
        });
        assert_eq!(plan.entries(), expected);
        assert!(!output.exists(), "planning creates nothing");
        Ok(())
    }

    #[test]
    fn a_single_file_lands_under_its_own_name() -> TestResult {
        let source = tempfile::tempdir()?;
        let output = tempfile::tempdir()?;
        tree(source.path())?;
        let file = source.path().join("a");

        let plan = ExportTree::plan(&file, output.path())?;

        assert_eq!(
            plan.entries(),
            [ExportEntry {
                source: file.canonicalize()?,
                target: output.path().canonicalize()?.join("a"),
            }]
        );
        Ok(())
    }

    #[test]
    fn an_output_inside_the_source_is_refused_and_not_created() -> TestResult {
        let source = tempfile::tempdir()?;
        tree(source.path())?;
        let output = source.path().join("export");

        let error = ExportTree::plan(source.path(), &output)
            .expect_err("invariant: an output inside the source overlaps it");

        assert!(matches!(error, ExportTreeError::OutputOverlapsInput { .. }));
        assert!(!output.exists());
        Ok(())
    }

    #[test]
    fn an_output_containing_or_equal_to_the_source_is_refused() -> TestResult {
        let parent = tempfile::tempdir()?;
        let source = parent.path().join("study");
        tree(&source)?;

        for output in [parent.path(), source.as_path()] {
            let error = ExportTree::plan(&source, output)
                .expect_err("invariant: the output contains or is the source");
            assert!(matches!(error, ExportTreeError::OutputOverlapsInput { .. }));
        }
        Ok(())
    }

    #[test]
    fn a_non_empty_output_is_refused() -> TestResult {
        let source = tempfile::tempdir()?;
        let output = tempfile::tempdir()?;
        tree(source.path())?;
        std::fs::write(output.path().join("keep"), "keep")?;

        let error = ExportTree::plan(source.path(), output.path())
            .expect_err("invariant: the output folder holds a file");

        let expected = output.path().canonicalize()?;
        assert!(matches!(error, ExportTreeError::OutputNotEmpty(path) if path == expected));
        Ok(())
    }

    #[test]
    fn a_missing_source_names_the_failed_step() -> TestResult {
        let output = tempfile::tempdir()?;
        let missing = output.path().join("absent");

        let error = ExportTree::plan(&missing, output.path())
            .expect_err("invariant: the source does not exist");

        assert!(matches!(
            error,
            ExportTreeError::Io { action: "resolve", path, .. } if path == missing
        ));
        Ok(())
    }
}
