pub mod slice_tile;
pub mod pad_crop;
pub mod shape_axes;
pub mod sources;

pub use slice_tile::{slice_image, checker_board, tile, join_series, paste};
pub use pad_crop::{constant_pad, mirror_pad, wrap_pad, zero_flux_neumann_pad, fft_pad, crop, region_of_interest};
pub use shape_axes::{flip, shrink, expand, cyclic_shift, permute_axes, dicom_orient};
pub use sources::{gaussian_image_source, grid_image_source, gabor_image_source};
