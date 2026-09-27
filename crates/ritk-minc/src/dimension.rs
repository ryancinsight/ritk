//! MINC2 dimension metadata: HDF5 paths, axis names, and the parsed
//! per-dimension record.

/// MINC2 HDF5 path to the dimensions group.
pub const DIMENSIONS_PATH: &str = "minc-2.0/dimensions";

/// MINC2 HDF5 path to the image dataset.
pub const IMAGE_PATH: &str = "minc-2.0/image/0/image";

/// Recognized spatial dimension names in canonical order (x, y, z).
pub const SPATIAL_DIM_NAMES: [&str; 3] = ["xspace", "yspace", "zspace"];

/// Parsed metadata for a single MINC2 spatial dimension.
#[derive(Debug, Clone)]
pub struct MincDimension {
    /// Dimension name (e.g. "xspace", "yspace", "zspace").
    pub name: String,
    /// Physical start coordinate in mm.
    pub start: f64,
    /// Voxel spacing in mm.
    pub step: f64,
    /// Number of voxels along this axis.
    pub length: usize,
    /// Direction cosine vector (3 components).
    pub direction_cosines: [f64; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minc_dimension_fields_accessible() {
        let dim = MincDimension {
            name: String::from("xspace"),
            start: -64.0,
            step: 1.0,
            length: 128,
            direction_cosines: [1.0, 0.0, 0.0],
        };
        assert_eq!(dim.name, "xspace");
        assert!((dim.start - (-64.0)).abs() < 1e-10);
        assert!((dim.step - 1.0).abs() < 1e-10);
        assert_eq!(dim.length, 128);
        assert_eq!(dim.direction_cosines, [1.0, 0.0, 0.0]);
    }
}
