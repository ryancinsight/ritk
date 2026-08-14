//! Tests for the `tract` command group.
//!
//! These cover the command's own responsibilities — seeding, the frame the
//! streamlines are written in, and argument validation. Tracking behaviour is
//! `ritk_tractography`'s contract and the direction lookup is
//! `ritk_diffusion::maps::DtiVolume`'s; both are tested there.

use super::*;

/// A `[2, 2, 2]` volume: eight voxels, so `plane` is 4 and `columns` is 2.
const SHAPE: [usize; 3] = [2, 2, 2];

#[test]
fn seeding_selects_only_voxels_at_or_above_the_floor() {
    // The floor is inclusive: a voxel exactly at the threshold is white matter
    // by the same convention that admits everything above it.
    let anisotropy = [0.0, 0.1, 0.25, 0.9, 0.0, 0.0, 0.0, 0.0];
    let seeds = seed(&anisotropy, SHAPE, 0.25, 0);
    assert_eq!(seeds.len(), 2, "voxels 2 and 3 qualify");
}

#[test]
fn seed_indices_decompose_in_depth_row_column_order() {
    // Voxel 5 of a [2, 2, 2] volume is depth 1, row 0, column 1. Getting this
    // wrong seeds a different voxel than the one whose anisotropy was checked,
    // and nothing downstream would reveal it.
    let mut anisotropy = [0.0_f64; 8];
    anisotropy[5] = 0.9;

    let seeds = seed(&anisotropy, SHAPE, 0.25, 0);
    assert_eq!(seeds.len(), 1);
    let index = seeds[0].to_array();
    assert_eq!(
        index,
        [1.0, 0.0, 1.0],
        "voxel 5 is depth 1, row 0, column 1"
    );
}

#[test]
fn a_seed_cap_thins_the_volume_evenly_rather_than_truncating() {
    // Truncating would seed only whichever end of the volume is stored first,
    // producing a tractogram that covers half a brain and looks like a tracking
    // failure rather than a seeding choice.
    let anisotropy = [0.9_f64; 8];
    let seeds = seed(&anisotropy, SHAPE, 0.25, 4);

    assert!(
        seeds.len() <= 4,
        "the cap is respected, got {}",
        seeds.len()
    );
    let depths: Vec<f64> = seeds.iter().map(|point| point.to_array()[0]).collect();
    assert!(
        depths.contains(&0.0) && depths.contains(&1.0),
        "seeds must span both slices, got depths {depths:?}"
    );
}

#[test]
fn a_zero_cap_seeds_every_qualifying_voxel() {
    let anisotropy = [0.9_f64; 8];
    assert_eq!(seed(&anisotropy, SHAPE, 0.25, 0).len(), 8);
}

#[test]
fn nothing_qualifies_below_the_floor() {
    // The caller turns this into an error naming the peak, rather than writing
    // an empty tractogram that reads as a successful run.
    let anisotropy = [0.1_f64; 8];
    assert!(seed(&anisotropy, SHAPE, 0.25, 0).is_empty());
}

// ── Output format ─────────────────────────────────────────────────────────────

#[test]
fn the_format_follows_the_extension() {
    for (name, expected) in [
        ("tracks.tck", TrackFormat::Tck),
        ("tracks.trk", TrackFormat::Trk),
        ("tracks.trx", TrackFormat::Trx),
    ] {
        assert_eq!(TrackFormat::from_path(Path::new(name)), Some(expected));
    }
}

#[test]
fn extension_matching_ignores_case() {
    // A path from a Windows shell or a GUI file picker often arrives uppercased,
    // and rejecting it would look like an unsupported format rather than a
    // spelling difference.
    assert_eq!(
        TrackFormat::from_path(Path::new("TRACKS.TCK")),
        Some(TrackFormat::Tck)
    );
}

#[test]
fn an_unknown_extension_is_rejected_rather_than_guessed() {
    // Defaulting to one format would write a file whose contents do not match
    // its name, which nothing downstream would flag.
    for name in ["tracks.vtk", "tracks.nii.gz", "tracks", "tracks."] {
        assert_eq!(
            TrackFormat::from_path(Path::new(name)),
            None,
            "{name} should not resolve to a format"
        );
    }
}
