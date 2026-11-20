//! Build an owned dataset from validated scan events.

mod materialization;
mod primitive;

pub(crate) use materialization::materialize;
