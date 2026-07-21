use super::*;

/// Every layout must report at least one viewport id.
#[test]
fn test_layout_mode_viewport_ids_non_empty() {
    for mode in LayoutMode::all() {
        let ids = mode.viewport_ids();
        assert!(
            !ids.is_empty(),
            "{:?}.viewport_ids() must return at least one id",
            mode
        );
    }
}

/// The viewport ids returned by each layout must be pairwise distinct.
#[test]
fn test_layout_mode_viewport_ids_distinct() {
    for mode in LayoutMode::all() {
        let ids = mode.viewport_ids();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(
                    ids[i], ids[j],
                    "{:?}.viewport_ids() has duplicate at [{i}] and [{j}]",
                    mode
                );
            }
        }
    }
}

/// `TwoByTwo` must have exactly four viewport ids.
#[test]
fn test_two_by_two_has_four_ids() {
    let ids = LayoutMode::TwoByTwo.viewport_ids();
    assert_eq!(
        ids.len(),
        4,
        "TwoByTwo must have exactly 4 viewport ids, got {}",
        ids.len()
    );
}

/// `Single` must have exactly one viewport id: `Main`.
#[test]
fn test_single_has_main_id() {
    let ids = LayoutMode::Single.viewport_ids();
    assert_eq!(ids.len(), 1, "Single must have exactly 1 viewport id");
    assert_eq!(ids[0], ViewportId::Main, "Single's only id must be Main");
}

/// `all()` must enumerate every variant exactly once.
#[test]
fn test_layout_mode_all_complete() {
    let all = LayoutMode::all();
    assert_eq!(all.len(), 5, "LayoutMode::all() must list all 5 variants");
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            assert_ne!(
                all[i], all[j],
                "LayoutMode::all() has duplicate at [{i}] and [{j}]"
            );
        }
    }
}

/// `label()` must return a non-empty string for every variant.
#[test]
fn test_layout_mode_labels_non_empty() {
    for mode in LayoutMode::all() {
        assert!(
            !mode.label().is_empty(),
            "{:?}.label() must not be empty",
            mode
        );
    }
}

/// Serde round-trip: every variant must survive JSON serialisation.
#[test]
fn test_layout_mode_serde_round_trip() {
    for &mode in LayoutMode::all() {
        let json = serde_json::to_string(&mode).expect("infallible: validated precondition");
        let recovered: LayoutMode =
            serde_json::from_str(&json).expect("infallible: validated precondition");
        assert_eq!(
            mode, recovered,
            "{:?} serde round-trip must preserve value",
            mode
        );
    }
}
