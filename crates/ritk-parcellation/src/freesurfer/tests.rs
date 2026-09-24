//! Hostile input: every reader returns a typed result for arbitrary bytes.
//!
//! The property is panic-freedom — a panic fails the test, whatever the
//! result — over two input families: raw bytes, and bytes behind each format's
//! valid magic so the search reaches past the first check into the counts and
//! records. Reading from an in-memory slice also bounds each case's work by
//! its length, so a forged count cannot turn a case into a long allocation.

use proptest::prelude::*;

use super::*;

/// Read `bytes` as every format; whatever is accepted must hold its invariants.
fn read_every_format(bytes: &[u8]) {
    if let Ok(surface) = Surface::read(bytes) {
        let vertices = surface.vertex_count();
        assert!(
            surface
                .faces()
                .iter()
                .flatten()
                .all(|v| (*v as usize) < vertices)
        );
        assert!(surface.vertices().iter().flatten().all(|c| c.is_finite()));
    }
    if let Ok(data) = Morphometry::read(bytes) {
        // Header is 15 bytes; each value is 4 bytes of real input.
        assert!(data.values().len() * 4 + 15 <= bytes.len());
    }
    if let Ok(annotation) = SurfaceAnnotation::read(bytes) {
        let table = annotation.color_table();
        assert!(
            annotation
                .vertex_labels()
                .iter()
                .all(|label| *label == crate::BACKGROUND || table.get_label(*label).is_some())
        );
    }
    if let Ok(label) = SurfaceLabel::read(bytes) {
        assert!(label.vertices().iter().all(|point| point.value.is_finite()));
    }
    if let Ok(table) = lut::read(bytes) {
        assert!(!table.is_empty());
        let labels: std::collections::HashSet<u32> =
            table.entries().iter().map(|entry| entry.id.0).collect();
        assert_eq!(labels.len(), table.len(), "labels are unique");
    }
}

proptest! {
    #[test]
    fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
        read_every_format(&bytes);
    }

    #[test]
    fn bytes_behind_a_valid_magic_never_panic(
        magic in prop_oneof![Just([0xFF_u8, 0xFF, 0xFE]), Just([0xFF_u8, 0xFF, 0xFF])],
        body in proptest::collection::vec(any::<u8>(), 0..512),
    ) {
        let mut bytes = magic.to_vec();
        bytes.extend_from_slice(&body);
        read_every_format(&bytes);
    }

    /// Small big-endian integers make counts plausible, which is what drives a
    /// reader deep into its record loops.
    #[test]
    fn small_integer_fields_never_panic(
        fields in proptest::collection::vec(-3_i32..300, 0..64),
    ) {
        let bytes: Vec<u8> = fields.iter().flat_map(|field| field.to_be_bytes()).collect();
        read_every_format(&bytes);
        let mut behind_magic = vec![0xFF, 0xFF, 0xFF];
        behind_magic.extend_from_slice(&bytes);
        read_every_format(&behind_magic);
    }

    #[test]
    fn arbitrary_text_never_panics(text in "[-0-9a-z# .\n]{0,256}") {
        read_every_format(text.as_bytes());
    }
}
