//! NRRD reader entry points and focused decoding helpers.

mod decode;
mod diffusion;
mod header;
mod stored;
mod volume;

pub use diffusion::read_nrrd_gradient_scheme;
pub use header::{
    read_nrrd_header, read_nrrd_header_map, NrrdHeader, NrrdHeaderError, NrrdKeyValueRecord,
};
pub(crate) use header::{is_standard_field_name, MAX_HEADER_BYTES, MAX_HEADER_ENTRIES};
pub(crate) use stored::{has_diffusion_metadata, NrrdReadSession};
pub use stored::{
    read_nrrd_stored, read_nrrd_stored_series, NrrdSpatialMetadataField, NrrdStoredReadError,
};
pub(crate) use volume::NrrdReadPlan;
pub use volume::{read_nrrd, read_nrrd_series, NrrdReader};
