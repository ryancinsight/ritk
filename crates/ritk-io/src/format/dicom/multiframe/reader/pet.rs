//! Temporal and gated Enhanced PET volume decoding.

use std::{path::Path, sync::Arc};

use anyhow::{ensure, Context, Result};
use dicom::object::DefaultDicomObject;
use ritk_dicom::{parse_bytes_with_budget, read_file_with_budget, DicomRsBackend};
use ritk_spatial::{Direction, Point, Spacing};

use super::{
    concatenation::order_members,
    frame_source::FrameSource,
    volume::{decode_spatial_frames, MultiFrameVolume},
};
use crate::format::dicom::{
    identity::{optional_uid, CONCATENATION_UID},
    multiframe::{
        pet_dimensions::{frame_groups, is_enhanced_pet, PetFrameGroup},
        BodyWeightSuvVolume, EnhancedPetDimension,
    },
    reader::{scan_dicom_path_with_budget, DicomReadBudget, DicomSliceMetadata},
    SliceProvenance,
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
    /// Per-plane source for a resampled reconstruction.
    pub slice_provenance: Option<Vec<SliceProvenance>>,
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

/// Decode a standalone Enhanced PET object or discover a selected member's concatenation.
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
    let bytes = read_file_with_budget(path, &budget.parser())?;
    let mut retained = 0;
    charge_retained_bytes(&mut retained, bytes.len(), &budget)?;
    let obj = parse_bytes_with_budget::<DicomRsBackend>(&bytes, &budget.parser())?;
    drop(bytes);
    if optional_uid(&obj, CONCATENATION_UID)?.is_some() {
        drop(obj);
        let series = scan_dicom_path_with_budget(path, &budget)?;
        return load_enhanced_pet_volumes_from_instances(&series.metadata.slices, &budget);
    }
    decode_pet(&[(path, obj)], &budget)
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
    let mut retained = 0;
    charge_retained_bytes(&mut retained, bytes.len(), budget)?;
    let obj = parse_bytes_with_budget::<DicomRsBackend>(bytes, &budget.parser())?;
    decode_pet(&[(name.as_ref(), obj)], budget)
}

/// Decode the spatial volumes from the exact scanned members of one
/// Enhanced PET acquisition. Member order is established by DICOM frame offsets.
/// If Part 10 bytes are present, they take precedence over the path.
/// # Errors
/// Rejects mixed acquisitions, missing or overlapping known members, conflicting
/// dimension definitions, malformed pixels, and budget violations. A missing
/// final member is detectable only when a total is declared.
pub fn load_enhanced_pet_volumes_from_instances(
    instances: &[DicomSliceMetadata],
    budget: &DicomReadBudget,
) -> Result<Vec<EnhancedPetVolume>> {
    ensure!(
        !instances.is_empty() && instances.len() <= budget.max_instances(),
        "Invalid Enhanced PET instance count for read budget"
    );
    let mut retained = 0;
    let mut objects = Vec::with_capacity(instances.len());
    for instance in instances {
        let owned;
        let bytes = if let Some(bytes) = instance.part10_bytes.as_deref() {
            bytes
        } else {
            owned = read_file_with_budget(&instance.path, &budget.parser())?;
            &owned
        };
        charge_retained_bytes(&mut retained, bytes.len(), budget)?;
        let object = parse_bytes_with_budget::<DicomRsBackend>(bytes, &budget.parser())
            .with_context(|| format!("Failed to parse Enhanced PET member {:?}", instance.path))?;
        objects.push((instance.path.as_path(), object));
    }
    decode_pet(&objects, budget)
}

fn charge_retained_bytes(total: &mut usize, bytes: usize, budget: &DicomReadBudget) -> Result<()> {
    *total = total
        .checked_add(bytes)
        .context("Enhanced PET retained byte count overflows usize")?;
    ensure!(
        *total <= budget.max_retained_bytes(),
        "Enhanced PET exceeds retained-byte budget"
    );
    Ok(())
}

fn pet_source<'a>(
    path: &'a Path,
    obj: &'a DefaultDicomObject,
    budget: &DicomReadBudget,
) -> Result<FrameSource<'a>> {
    ensure!(
        is_enhanced_pet(&obj.meta().media_storage_sop_class_uid),
        "DICOM object {path:?} is not Enhanced PET Image Storage"
    );
    FrameSource::read(path, obj, budget)
}

pub(super) fn decode_single_spatial_pet(
    path: &Path,
    obj: &DefaultDicomObject,
    budget: &DicomReadBudget,
) -> Result<MultiFrameVolume> {
    let sources = [pet_source(path, obj, budget)?];
    let members = order_members(&sources)?;
    let groups = frame_groups(&sources, &members)?;
    let [group] = groups.as_slice() else {
        anyhow::bail!(
            "Enhanced PET has multiple temporal or gated volumes; use load_enhanced_pet_volumes"
        );
    };
    decode_spatial_frames(&sources, budget, Some(&group.frames))
}

fn decode_pet(
    objects: &[(&Path, DefaultDicomObject)],
    budget: &DicomReadBudget,
) -> Result<Vec<EnhancedPetVolume>> {
    let mut original_bytes = 0usize;
    let sources = objects
        .iter()
        .map(|(path, obj)| {
            let instance = pet_source(path, obj, budget)?;
            let info = &instance.info;
            original_bytes = info
                .rows
                .checked_mul(info.cols)
                .and_then(|pixels| pixels.checked_mul(info.n_frames))
                .and_then(|pixels| pixels.checked_mul(2 * size_of::<f32>()))
                .and_then(|bytes| original_bytes.checked_add(bytes))
                .context("Enhanced PET decoded size overflows usize")?;
            budget.checked_decoded_bytes(original_bytes)?;
            Ok(instance)
        })
        .collect::<Result<Vec<_>>>()?;
    let members = order_members(&sources)?;
    let groups = frame_groups(&sources, &members)?;
    let mut volumes = Vec::new();
    let mut remaining = budget.max_decoded_bytes();
    for PetFrameGroup { dimensions, frames } in groups {
        let volume_budget =
            DicomReadBudget::try_new(budget.parser(), budget.max_retained_bytes(), remaining)?;
        let spatial = decode_spatial_frames(&sources, &volume_budget, Some(&frames))?;
        // Charge both carriers conservatively even if this volume has no valid SUV.
        let bytes = spatial
            .data
            .len()
            .checked_mul(2 * size_of::<f32>())
            .and_then(|bytes| {
                let planes = spatial.slice_provenance.as_ref().map_or(0, Vec::len);
                bytes.checked_add(planes.checked_mul(size_of::<SliceProvenance>())?)
            })
            .context("Enhanced PET decoded volume size overflows usize")?;
        remaining = remaining
            .checked_sub(bytes)
            .context("Enhanced PET volume set exceeds decoded-byte budget")?;
        volumes.push(EnhancedPetVolume {
            dimensions,
            data: Arc::new(spatial.data),
            suv_bw: spatial.suv_bw,
            slice_provenance: spatial.slice_provenance,
            shape: spatial.shape,
            origin: spatial.origin,
            spacing: spatial.spacing,
            direction: spatial.direction,
        });
    }
    Ok(volumes)
}
