//! Tests for the `tract` command group.
//!
//! These cover the command's own responsibilities — output format selection
//! and argument validation. Seeding and tracking are tested at their reusable
//! `ritk_tractography` boundary.

use super::*;

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
