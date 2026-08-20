use super::*;

/// A 2×2×2 volume with three regions in a diagonal pattern.
///
/// ```text
/// z=0:   [1, 0]     z=1:   [0, 2]
///        [0, 3]            [3, 0]
/// ```
fn three_region_cube() -> Parcellation {
    let grid = ParcellationGrid::axis_aligned([2, 2, 2], [2.0; 3], [0.0; 3]).expect("valid grid");
    Parcellation::new(
        Box::new([1, 0, 0, 3, 0, 2, 3, 0]),
        grid,
        vec![
            (1, "Region A".into()),
            (2, "Region B".into()),
            (3, "Region C".into()),
        ],
    )
    .expect("valid parcellation")
}

fn grid_2x2x2() -> ParcellationGrid {
    ParcellationGrid::axis_aligned([2, 2, 2], [2.0; 3], [0.0; 3]).expect("valid grid")
}

// ── Construction ─────────────────────────────────────────────────────────

#[test]
fn a_label_array_that_does_not_cover_the_grid_is_rejected() {
    let error = Parcellation::new(Box::new([1, 2, 3]), grid_2x2x2(), Vec::new()).unwrap_err();
    match error {
        ParcellationError::LabelCountMismatch { expected, actual } => {
            assert_eq!(expected, 8);
            assert_eq!(actual, 3);
        }
        other => panic!("expected a count mismatch, got {other:?}"),
    }
}

/// An all-background volume answers no question that can be asked of a
/// parcellation, so it is rejected where it is built rather than returning empty
/// results at every call site.
#[test]
fn an_all_background_volume_is_rejected() {
    let error = Parcellation::new(
        vec![BACKGROUND; 8].into_boxed_slice(),
        grid_2x2x2(),
        Vec::new(),
    )
    .unwrap_err();
    assert!(matches!(error, ParcellationError::EmptyParcellation));
}

#[test]
fn region_labels_are_sorted_deduplicated_and_exclude_background() {
    let parcellation = three_region_cube();
    assert_eq!(parcellation.region_labels(), vec![1, 2, 3]);
    assert_eq!(parcellation.region_count(), 3);
    assert!(parcellation.contains_region(2));
    assert!(!parcellation.contains_region(BACKGROUND));
    assert!(!parcellation.contains_region(99));
}

// ── Names ────────────────────────────────────────────────────────────────

#[test]
fn region_names_resolve_by_label() {
    let parcellation = three_region_cube();
    assert_eq!(parcellation.name_of(1), Some("Region A"));
    assert_eq!(parcellation.name_of(3), Some("Region C"));
    assert_eq!(parcellation.name_of(99), None);
}

/// Names arrive from a lookup table in whatever order the file lists them, and
/// resolution is by binary search, so construction must sort them. An unsorted
/// table would make lookups silently miss.
#[test]
fn names_supplied_out_of_order_still_resolve() {
    let parcellation = Parcellation::new(
        Box::new([1, 0, 0, 3, 0, 2, 3, 0]),
        grid_2x2x2(),
        vec![
            (3, "Third".into()),
            (1, "First".into()),
            (2, "Second".into()),
        ],
    )
    .expect("valid parcellation");

    assert_eq!(parcellation.name_of(1), Some("First"));
    assert_eq!(parcellation.name_of(2), Some("Second"));
    assert_eq!(parcellation.name_of(3), Some("Third"));
}

// ── Lookup ───────────────────────────────────────────────────────────────

#[test]
fn labels_resolve_at_voxel_centres() {
    let parcellation = three_region_cube();
    assert_eq!(parcellation.label_at(&Point::new([0.0, 0.0, 0.0])), Some(1));
    assert_eq!(parcellation.label_at(&Point::new([0.0, 2.0, 0.0])), Some(0));
    assert_eq!(parcellation.label_at(&Point::new([2.0, 2.0, 0.0])), Some(3));
    assert_eq!(parcellation.label_at(&Point::new([2.0, 0.0, 2.0])), Some(2));
}

/// "Outside the volume" and "inside but unlabelled" are different claims, and a
/// caller counting skipped streamlines needs to tell them apart.
#[test]
fn outside_the_volume_is_distinct_from_background_inside_it() {
    let parcellation = three_region_cube();
    assert_eq!(
        parcellation.label_at(&Point::new([0.0, 2.0, 0.0])),
        Some(BACKGROUND)
    );
    assert_eq!(parcellation.label_at(&Point::new([100.0, 0.0, 0.0])), None);
}

#[test]
fn an_index_outside_the_grid_has_no_label() {
    let parcellation = three_region_cube();
    assert_eq!(parcellation.label_at_index([0, 0, 0]), Some(1));
    assert_eq!(parcellation.label_at_index([2, 0, 0]), None);
}

// ── Remapping ────────────────────────────────────────────────────────────

/// Merging fine parcels into coarse ones is how an atlas is coarsened, and the
/// operation has to preserve background rather than folding it into whatever the
/// mapping returns for zero.
#[test]
fn remapping_merges_regions_and_leaves_background_alone() {
    let parcellation = three_region_cube();
    let merged = parcellation
        .remap_labels(
            |label| if label == 3 { 1 } else { label },
            vec![(1, "Merged".into()), (2, "Region B".into())],
        )
        .expect("merge leaves regions");

    assert_eq!(merged.region_labels(), vec![1, 2]);
    assert_eq!(merged.name_of(1), Some("Merged"));
    // The voxels that were region 3 now read region 1.
    assert_eq!(merged.label_at(&Point::new([2.0, 2.0, 0.0])), Some(1));
    // Background voxels are untouched.
    assert_eq!(
        merged.label_at(&Point::new([0.0, 2.0, 0.0])),
        Some(BACKGROUND)
    );
}

#[test]
fn remapping_everything_to_background_is_rejected() {
    let parcellation = three_region_cube();
    let error = parcellation
        .remap_labels(|_| BACKGROUND, Vec::new())
        .unwrap_err();
    assert!(matches!(error, ParcellationError::EmptyParcellation));
}

/// Restricting to a subset is the common preparation for a targeted connectome —
/// only cortical parcels, say — and must drop both the voxels and the names of
/// everything else.
#[test]
fn retaining_a_subset_drops_the_other_regions_and_their_names() {
    let parcellation = three_region_cube();
    let cortical = parcellation
        .retain_regions(&[1, 2])
        .expect("regions remain");

    assert_eq!(cortical.region_labels(), vec![1, 2]);
    assert_eq!(cortical.name_of(3), None);
    assert_eq!(
        cortical.label_at(&Point::new([2.0, 2.0, 0.0])),
        Some(BACKGROUND)
    );
    // The grid is carried through unchanged, so points still resolve the same way.
    assert_eq!(cortical.grid(), parcellation.grid());
}

#[test]
fn retaining_only_absent_regions_is_rejected() {
    let parcellation = three_region_cube();
    let error = parcellation.retain_regions(&[42]).unwrap_err();
    assert!(matches!(error, ParcellationError::EmptyParcellation));
}

// ── Serialisation ────────────────────────────────────────────────────────

/// The grid's cached inverse is derived state; a round trip must reconstruct a
/// parcellation whose lookups agree, not merely one whose fields compare equal.
#[test]
fn a_serde_round_trip_preserves_lookups() {
    let parcellation = three_region_cube();
    let encoded = serde_json::to_string(&parcellation).expect("serialise");
    let decoded: Parcellation = serde_json::from_str(&encoded).expect("deserialise");

    assert_eq!(decoded.region_labels(), parcellation.region_labels());
    for offset in 0..parcellation.grid().voxel_count() {
        let index = parcellation
            .grid()
            .index_of_offset(offset)
            .expect("in-range offset");
        let point = parcellation.grid().physical_point_of(index);
        assert_eq!(
            decoded.label_at(&point),
            parcellation.label_at(&point),
            "voxel {index:?}"
        );
    }
}

/// A decoded value must pass the same checks a constructed one does.
///
/// `Deserialize` is a second construction path, and a derived one skips the
/// constructor entirely — so a document is free to declare a grid of one size
/// and a label array of another. The value that produced would index past the
/// end of its own labels on the first lookup.
#[test]
fn a_document_whose_labels_do_not_cover_its_grid_is_rejected() {
    let encoded = serde_json::to_string(&three_region_cube()).expect("serialise");
    // Drop the volume to three labels while the grid still declares eight.
    let truncated = encoded.replace("\"labels\":[1,0,0,3,0,2,3,0]", "\"labels\":[1,0,3]");
    assert_ne!(truncated, encoded, "the fixture must have been edited");

    let error = serde_json::from_str::<Parcellation>(&truncated).unwrap_err();
    assert!(
        error.to_string().contains("labels were supplied"),
        "expected the count-mismatch message, got {error}"
    );
}

/// The same for a document with no labelled voxel at all.
#[test]
fn an_all_background_document_is_rejected() {
    let encoded = serde_json::to_string(&three_region_cube()).expect("serialise");
    let emptied = encoded.replace(
        "\"labels\":[1,0,0,3,0,2,3,0]",
        "\"labels\":[0,0,0,0,0,0,0,0]",
    );
    assert_ne!(emptied, encoded, "the fixture must have been edited");

    let error = serde_json::from_str::<Parcellation>(&emptied).unwrap_err();
    assert!(
        error.to_string().contains("no labelled regions"),
        "expected the empty-parcellation message, got {error}"
    );
}

/// The grid's cached inverse is derived state and is recomputed on decode
/// rather than read, so a document cannot supply one that disagrees with its
/// own direction matrix.
#[test]
fn the_grid_inverse_is_not_taken_from_the_document() {
    let encoded = serde_json::to_string(&three_region_cube()).expect("serialise");
    assert!(
        !encoded.contains("inverse"),
        "derived state must stay off the wire: {encoded}"
    );

    let decoded: Parcellation = serde_json::from_str(&encoded).expect("round trip");
    assert_eq!(decoded.label_at(&Point::new([2.0, 2.0, 0.0])), Some(3));
}
