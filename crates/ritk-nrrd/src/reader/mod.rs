//! NRRD reader entry points and focused decoding helpers.

mod decode;
mod diffusion;
mod header;
mod volume;

pub use diffusion::read_nrrd_gradient_scheme;
pub use header::read_nrrd_header_map;
pub use volume::{read_nrrd, read_nrrd_series, NrrdReader};
