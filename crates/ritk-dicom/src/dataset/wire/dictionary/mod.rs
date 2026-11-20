//! Implicit VR lookup from the numeric PS3.6 registry.
//!
//! The pinned edition, checksum, and regeneration procedure are in
//! scripts/dicom-dictionary.py. Repeating-tag patterns preserve all fixed bits
//! (including the element number) and prefer the most specific match.

mod table;

pub(crate) use table::vr;
