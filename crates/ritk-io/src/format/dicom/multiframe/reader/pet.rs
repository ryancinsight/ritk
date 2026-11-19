//! Temporal and gated Enhanced PET volume decoding.

use std::{path::Path, sync::Arc};

use anyhow::{ensure, Context, Result};
use dicom::object::DefaultDicomObject;
use ritk_dicom::{parse_bytes_with_budget, parse_file_with_budget, DicomRsBackend};
use ritk_spatial::{Direction, Point, Spacing};

use super::{header::extract_multiframe_header, volume::decode_spatial_frames};
use crate::format::dicom::{
    multiframe::{
        per_frame::extract_functional_groups,
        pet_dimensions::{frame_groups, is_enhanced_pet, PetFrameGroup},
        types::PerFrameInfo,
        BodyWeightSuvVolume, EnhancedPetDimension,
    },
    reader::DicomReadBudget,
};

/// One spatial volume in a temporal or gated Enhanced PET object.
#[derive(Debug, Clone, PartialEq)]
pub struct EnhancedPetVolume {
    /// Nonspatial dimensions identifying the volume, in DICOM dimension order.
    pub dimensions: Vec<EnhancedPetDimension>,
    /// Display values in depth, row, column order.
    pub data: Arc<Vec<f32>>,
    /// Body-weight SUV aligned with the display voxel indices.
    pub suv_bw: Option<BodyWeightSuvVolume>,
    /// Volume shape in depth, row, column order.
    pub shape: [usize; 3],
    /// Patient-space position of the first voxel, in millimetres.
    pub origin: Point<3>,
    /// Depth, row, and column spacing in millimetres.
    pub spacing: Spacing<3>,
    /// Direction of the three storage axes in patient space.
    pub direction: Direction<3>,
}

/// Selection within a shared, independently reconstructed Enhanced PET volume set.
#[derive(Debug, Clone, PartialEq)]
pub struct EnhancedPetSelection {
    volumes: Arc<[EnhancedPetVolume]>,
    selected: usize,
}

impl EnhancedPetSelection {
    /// Select the first volume in a decoded set.
    pub fn new(volumes: Vec<EnhancedPetVolume>) -> Option<Self> {
        (!volumes.is_empty()).then(|| Self {
            volumes: volumes.into(),
            selected: 0,
        })
    }

    /// Spatial volumes ordered by their nonspatial dimension ordinals.
    pub fn volumes(&self) -> &[EnhancedPetVolume] {
        &self.volumes
    }

    /// Zero-based index of the current volume.
    pub const fn selected_index(&self) -> usize {
        self.selected
    }

    /// Choose a volume by index. An invalid index leaves the selection intact.
    pub fn select(&mut self, index: usize) -> Option<&EnhancedPetVolume> {
        let volume = self.volumes.get(index)?;
        self.selected = index;
        Some(volume)
    }
}

/// Decode the spatial volumes in one Enhanced PET Part 10 object.
///
/// Each time point, gate, and stack combination has its own calibrated volume.
/// Volumes appear in ascending DICOM dimension ordinal order.
///
/// # Errors
///
/// Returns an error for non-PET input, inconsistent dimensions, invalid
/// geometry or pixels, or a read budget violation.
pub fn load_enhanced_pet_volumes(path: impl AsRef<Path>) -> Result<Vec<EnhancedPetVolume>> {
    let path = path.as_ref();
    let budget = DicomReadBudget::DEFAULT;
    let obj = parse_file_with_budget::<DicomRsBackend, _>(path, &budget.parser())?;
    decode_pet(path, &obj, &budget)
}

/// Decode the spatial volumes in one named Enhanced PET byte payload.
///
/// The name identifies the object in diagnostics. The same structural and
/// decoded-byte budget applies to the complete object.
///
/// # Errors
///
/// Returns an error for non-PET input, inconsistent dimensions, invalid
/// geometry or pixels, or a read budget violation.
pub fn load_enhanced_pet_volumes_from_bytes(
    name: impl AsRef<Path>,
    bytes: &[u8],
    budget: &DicomReadBudget,
) -> Result<Vec<EnhancedPetVolume>> {
    let obj = parse_bytes_with_budget::<DicomRsBackend>(bytes, &budget.parser())?;
    decode_pet(name.as_ref(), &obj, budget)
}

pub(super) fn pet_spatial_groups(
    path: &Path,
    obj: &DefaultDicomObject,
    budget: &DicomReadBudget,
) -> Result<(Vec<PerFrameInfo>, Vec<PetFrameGroup>)> {
    ensure!(
        is_enhanced_pet(&obj.meta().media_storage_sop_class_uid),
        "DICOM object {path:?} is not Enhanced PET Image Storage"
    );
    let info = extract_multiframe_header(path, obj)?;
    ensure!(
        info.n_frames > 0 && info.rows > 0 && info.cols > 0,
        "Enhanced PET must contain nonempty frames"
    );
    let frame_bytes = info
        .rows
        .checked_mul(info.cols)
        .and_then(|px| px.checked_mul(2 * size_of::<f32>()))
        .context("Enhanced PET frame byte count overflows usize")?;
    let original_bytes = frame_bytes
        .checked_mul(info.n_frames)
        .context("Enhanced PET decoded size overflows usize")?;
    budget.checked_decoded_bytes(original_bytes)?;
    let geometry = extract_functional_groups(obj, info.n_frames);
    let groups = frame_groups(obj, path, info.n_frames, &geometry, info.image_orientation)?;
    let groups = groups.unwrap_or_else(|| {
        vec![PetFrameGroup {
            dimensions: Vec::new(),
            frames: (0..info.n_frames).collect(),
        }]
    });
    Ok((geometry, groups))
}

fn decode_pet(
    path: &Path,
    obj: &DefaultDicomObject,
    budget: &DicomReadBudget,
) -> Result<Vec<EnhancedPetVolume>> {
    let (geometry, groups) = pet_spatial_groups(path, obj, budget)?;
    let mut volumes = Vec::new();
    let mut remaining = budget.max_decoded_bytes();
    for PetFrameGroup { dimensions, frames } in groups {
        let volume_budget =
            DicomReadBudget::try_new(budget.parser(), budget.max_retained_bytes(), remaining)?;
        let spatial =
            decode_spatial_frames(path, obj, &volume_budget, Some(&frames), Some(&geometry))?;
        // Charge both carriers conservatively even if this volume has no valid SUV.
        let bytes = spatial
            .data
            .len()
            .checked_mul(2 * size_of::<f32>())
            .context("Enhanced PET decoded volume size overflows usize")?;
        remaining = remaining
            .checked_sub(bytes)
            .context("Enhanced PET volume set exceeds decoded-byte budget")?;
        volumes.push(EnhancedPetVolume {
            dimensions,
            data: Arc::new(spatial.data),
            suv_bw: spatial.suv_bw,
            shape: spatial.shape,
            origin: spatial.origin,
            spacing: spatial.spacing,
            direction: spatial.direction,
        });
    }
    Ok(volumes)
}
