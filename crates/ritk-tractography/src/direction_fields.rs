use ritk_spatial::{Point, Vector};

/// Build a tractography direction field from a single-voxel fODF.
///
/// Pre-extracts the strongest peak via [`FodField::find_peaks`] using
/// a 50×100 grid and 10% relative threshold.  Returns a closure that
/// always yields that peak direction, or `None` when no peak is found.
///
/// This is a convenience for bootstrapping FOD-based tractography from a
/// single-voxel CSD result.  Whole-brain tractography layers a spatial
/// neighbourhood lookup on top.
pub fn fod_peak_direction_field<'a>(
    fod: &'a ritk_diffusion::csd::FodField,
) -> impl Fn(&Point<3>) -> Option<Vector<3>> + 'a {
    let peaks = fod.find_peaks(50, 100, 0.1).unwrap_or_default();
    let strongest: Option<Vector<3>> = peaks.first().map(|peak| Vector::new(peak.direction));

    move |_point: &Point<3>| -> Option<Vector<3>> { strongest }
}

/// Build a tractography direction field from a single-voxel DTI result.
///
/// Returns a closure that always yields the principal eigenvector of the
/// fitted diffusion tensor, or `None` when the PEV is degenerate (e.g.
/// isotropic tensor with near-zero FA).
///
/// Whole-brain tractography layers a spatial neighbourhood lookup on top;
/// this convenience bootstraps single-voxel DTI-based streamline tracking.
pub fn dti_pev_direction_field<'a>(
    tensor: &'a ritk_diffusion::dti::DiffusionTensor,
) -> impl Fn(&Point<3>) -> Option<Vector<3>> + 'a {
    let pev = tensor.principal_eigenvector();
    let degenerate = tensor.fa() < 1e-10;
    let direction: Option<Vector<3>> = (!degenerate).then(|| Vector::new(pev));

    move |_point: &Point<3>| -> Option<Vector<3>> { direction }
}

/// Build a whole-brain tractography direction field from a CSD fODF volume.
///
/// At each integration step trilinear interpolation samples the fODF
/// coefficients from the 3-D spatial neighbourhood and extracts the
/// strongest peak direction via a 50×100 spherical-grid search with a 10%
/// relative-amplitude threshold.
///
/// This replaces the single-voxel [`fod_peak_direction_field`] with a
/// spatial neighbourhood lookup suitable for whole-brain streamline
/// tracking through a pre-computed fODF volume.
pub fn fod_volume_direction_field<'a>(
    volume: &'a ritk_diffusion::csd::FodVolume,
) -> impl Fn(&Point<3>) -> Option<Vector<3>> + 'a {
    move |point: &Point<3>| -> Option<Vector<3>> { volume.direction_at(point, 50, 100, 0.1) }
}

/// Build a whole-brain tractography direction field from a NODDI volume.
///
/// At each integration step a nearest-neighbour spatial lookup retrieves
/// the NODDI principal direction.  This is simpler than the CSD
/// [`fod_volume_direction_field`] because NODDI intrinsically yields a
/// single fibre orientation per voxel — no peak extraction is needed.
pub fn noddi_direction_field<'a>(
    volume: &'a ritk_diffusion::noddi::NoddiVolume,
) -> impl Fn(&Point<3>) -> Option<Vector<3>> + 'a {
    move |point: &Point<3>| -> Option<Vector<3>> { volume.direction_at(point) }
}
