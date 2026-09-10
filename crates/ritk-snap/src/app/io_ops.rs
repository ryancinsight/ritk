//! Filesystem-facing workflows for the application shell.
//!
//! Each submodule owns one workflow and keeps its methods on `SnapApp`, so
//! this file names the set and nothing else.

mod dialog;
mod export;
mod segmentation;
mod session;
