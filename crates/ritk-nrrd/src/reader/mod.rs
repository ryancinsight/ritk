//! NRRD reader entry points and focused decoding helpers.

mod decode;
mod diffusion;
mod header;
mod stored;
mod volume;

pub use diffusion::read_nrrd_gradient_scheme;
pub use header::{read_nrrd_header_map, NrrdHeaderError};
pub use stored::{
    read_nrrd_stored, read_nrrd_stored_series, NrrdSpatialMetadataField, NrrdStoredReadError,
};
pub use volume::{read_nrrd, read_nrrd_series, NrrdReader};
