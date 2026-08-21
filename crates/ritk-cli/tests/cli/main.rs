//! End-to-end exercises of the `ritk` binary.
//!
//! Every command's integration test is a module here rather than its own file
//! at `tests/`, because each such file is an independent binary that re-links
//! the whole stack. One harness links once.

mod parcellate_atlas;
mod tract_connectome;
