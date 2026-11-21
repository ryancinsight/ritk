use super::decimal_string::{format_dicom_decimal, format_pair, format_six, format_triplet};
use super::error::DicomWriteError;
use super::pixel_encoding::validate_spatial_metadata;

use aequitas::systems::si::{quantities::Length, units::Millimeter};
use anyhow::Result;
use eunomia::FloatElement;
use ritk_dicom::dataset::{Dataset, DicomTag, VR};

/// Patient-space geometry in RITK's [depth, row, column] axis order.
pub(super) struct SeriesGeometry {
    origin: [Length; 3],
    spacing: [Length; 3],
    direction_columns: [[f64; 3]; 3],
}

impl SeriesGeometry {
    pub(super) fn new(
        origin: [Length; 3],
        spacing: [Length; 3],
        direction_columns: [[f64; 3]; 3],
    ) -> Self {
        Self {
            origin,
            spacing,
            direction_columns,
        }
    }

    pub(super) fn validate(&self, depth: usize) -> Result<()> {
        let direction = std::array::from_fn(|index| self.direction_columns[index / 3][index % 3]);
        validate_spatial_metadata(
            &self.spacing.map(|length| length.in_unit::<Millimeter>()),
            &self.origin.map(|length| length.in_unit::<Millimeter>()),
            &direction,
        )?;
        for slice in 0..depth {
            if self
                .position_mm(slice)
                .into_iter()
                .any(|coordinate| !coordinate.is_finite())
            {
                return Err(DicomWriteError::InvalidSpatialMetadata.into());
            }
        }
        Ok(())
    }

    /// DICOM PS3.3 C.7.6.2 Image Plane Module. Pixel spacing is row then
    /// column, while orientation cosines are column then row.
    pub(super) fn emit_plane_tags(&self, dataset: &mut Dataset, slice: usize) -> Result<()> {
        let [_, [rx, ry, rz], [cx, cy, cz]] = self.direction_columns;
        let [depth, row, column] = self.spacing.map(|length| length.in_unit::<Millimeter>());
        dataset.put_value(
            DicomTag::new(0x0018, 0x0050),
            VR::DS,
            format_dicom_decimal(depth)?,
        );
        dataset.put_value(
            DicomTag::new(0x0020, 0x0032),
            VR::DS,
            format_triplet(self.position_mm(slice))?,
        );
        dataset.put_value(
            DicomTag::new(0x0020, 0x0037),
            VR::DS,
            format_six([cx, cy, cz, rx, ry, rz])?,
        );
        dataset.put_value(
            DicomTag::new(0x0028, 0x0030),
            VR::DS,
            format_pair([row, column])?,
        );
        Ok(())
    }

    /// Advance by `slice · Δdepth` along the depth axis, then encode the
    /// patient coordinates in millimetres at the DICOM boundary.
    fn position_mm(&self, slice: usize) -> [f64; 3] {
        let distance = self.spacing[0] * f64::from_count(slice);
        let normal = self.direction_columns[0];
        std::array::from_fn(|axis| {
            (self.origin[axis] + distance * normal[axis]).in_unit::<Millimeter>()
        })
    }
}
