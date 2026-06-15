use super::*;

/// `all()` must enumerate every variant exactly once with no duplicates.
///
/// Correctness criterion: `all().len() == 9` and pairwise distinctness.
#[test]
fn test_tool_kind_all_complete_and_distinct() {
    let all = ToolKind::all();
    assert_eq!(
        all.len(),
        11,
        "ToolKind::all() must contain all 11 variants, found {}",
        all.len()
    );
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            assert_ne!(
                all[i], all[j],
                "ToolKind::all() contains duplicate at indices {i} and {j}: {:?}",
                all[i]
            );
        }
    }
}

/// `label()` must return a non-empty string for every variant.
#[test]
fn test_tool_kind_label_non_empty() {
    for tool in ToolKind::all() {
        let label = tool.label();
        assert!(!label.is_empty(), "{:?}.label() must not be empty", tool);
    }
}

/// `tooltip()` must return a non-empty string for every variant, and must
/// be distinct from `label()` (a tooltip adds information beyond the label).
#[test]
fn test_tool_kind_tooltip_non_empty_and_distinct_from_label() {
    for tool in ToolKind::all() {
        let tooltip = tool.tooltip();
        assert!(
            !tooltip.is_empty(),
            "{:?}.tooltip() must not be empty",
            tool
        );
        assert_ne!(
            tooltip,
            tool.label(),
            "{:?}.tooltip() must differ from label()",
            tool
        );
    }
}

/// `icon()` must return a non-empty string for every variant.
#[test]
fn test_tool_kind_icon_non_empty() {
    for tool in ToolKind::all() {
        let icon = tool.icon();
        assert!(!icon.is_empty(), "{:?}.icon() must not be empty", tool);
    }
}

/// Serde round-trip: every variant must survive JSON serialisation and
/// deserialisation with value equality.
#[test]
fn test_tool_kind_serde_round_trip() {
    for &tool in ToolKind::all() {
        let json = serde_json::to_string(&tool)
            .unwrap_or_else(|e| panic!("{tool:?} serde_json::to_string failed: {e}"));
        let recovered: ToolKind = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("{tool:?} serde_json::from_str failed: {e}"));
        assert_eq!(
            tool, recovered,
            "{:?} serde round-trip must preserve value equality",
            tool
        );
    }
}

/// `ToolKind::Pan` must be the first element (toolbar primary position).
#[test]
fn test_tool_kind_pan_is_first() {
    assert_eq!(
        ToolKind::all()[0],
        ToolKind::Pan,
        "Pan must be the first tool in ToolKind::all()"
    );
}
