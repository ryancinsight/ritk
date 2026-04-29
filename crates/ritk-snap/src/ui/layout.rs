//! Viewer viewport layout modes and slot identifiers.
//!
//! # Mathematical specification
//!
//! A [`LayoutMode`] partitions the central panel rectangle into N non-overlapping
//! sub-rectangles.  Each sub-rectangle is identified by a [`ViewportId`].
//!
//! ## Layout invariants
//! - The union of all viewport rectangles equals the available panel area.
//! - No two viewport rectangles overlap (zero-width borders are acceptable).
//! - [`LayoutMode::viewport_ids`] returns exactly the set of [`ViewportId`]s
//!   active in that layout; the length of the slice equals N for that layout.
//!
//! ## Viewport id → axis mapping (conventional)
//! | ViewportId   | MPR axis |
//! |--------------|----------|
//! | TopLeft      | Axial    |
//! | TopRight     | Coronal  |
//! | BottomLeft   | Sagittal |
//! | BottomRight  | 3-D/MIP  |
//! | Main         | Axial    |
//! | Left         | Axial    |
//! | Center       | Coronal  |
//! | Right        | Sagittal |

/// Viewer viewport layout modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum LayoutMode {
    /// Single viewport occupying the full central panel area.
    Single,
    /// 2×2 grid: top-left = axial, top-right = coronal,
    /// bottom-left = sagittal, bottom-right = 3-D/MIP.
    TwoByTwo,
    /// One large main view on the left plus three smaller views stacked on
    /// the right (1 + 3 layout).
    OneMainThreeSmall,
    /// Two side-by-side viewports for direct image comparison.
    SideBySide,
    /// Three horizontal viewports: axial | coronal | sagittal.
    ThreeHorizontal,
}

/// Identity token for a viewport slot within the active layout.
///
/// Not every variant is active in every layout; [`LayoutMode::viewport_ids`]
/// returns the active subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewportId {
    /// The sole viewport in [`LayoutMode::Single`]; large left pane in
    /// [`LayoutMode::OneMainThreeSmall`].
    Main,
    /// Top-left cell of the 2×2 grid (axial).
    TopLeft,
    /// Top-right cell of the 2×2 grid (coronal).
    TopRight,
    /// Bottom-left cell of the 2×2 grid (sagittal).
    BottomLeft,
    /// Bottom-right cell of the 2×2 grid (3-D/MIP).
    BottomRight,
    /// Left pane of a side-by-side or three-horizontal layout.
    Left,
    /// Centre pane of a three-horizontal layout.
    Center,
    /// Right pane of a side-by-side or three-horizontal layout.
    Right,
}

impl LayoutMode {
    /// The [`ViewportId`]s that are active (rendered) in this layout,
    /// in stable display order.
    ///
    /// # Invariants
    /// - The returned slice is non-empty.
    /// - All elements are distinct.
    /// - Length equals the number of viewport cells in the layout.
    pub fn viewport_ids(&self) -> &'static [ViewportId] {
        match self {
            LayoutMode::Single => &[ViewportId::Main],
            LayoutMode::TwoByTwo => &[
                ViewportId::TopLeft,
                ViewportId::TopRight,
                ViewportId::BottomLeft,
                ViewportId::BottomRight,
            ],
            LayoutMode::OneMainThreeSmall => &[
                ViewportId::Main,
                ViewportId::TopRight,
                ViewportId::Center,
                ViewportId::BottomRight,
            ],
            LayoutMode::SideBySide => &[ViewportId::Left, ViewportId::Right],
            LayoutMode::ThreeHorizontal => {
                &[ViewportId::Left, ViewportId::Center, ViewportId::Right]
            }
        }
    }

    /// Human-readable label used in the layout picker.
    pub fn label(&self) -> &'static str {
        match self {
            LayoutMode::Single => "Single",
            LayoutMode::TwoByTwo => "2×2",
            LayoutMode::OneMainThreeSmall => "1+3",
            LayoutMode::SideBySide => "Side-by-Side",
            LayoutMode::ThreeHorizontal => "3H",
        }
    }

    /// All layout modes in display order.
    ///
    /// Every variant appears exactly once; this slice is the single source of
    /// truth for UI iteration and serialisation round-trips.
    pub fn all() -> &'static [LayoutMode] {
        &[
            LayoutMode::Single,
            LayoutMode::TwoByTwo,
            LayoutMode::OneMainThreeSmall,
            LayoutMode::SideBySide,
            LayoutMode::ThreeHorizontal,
        ]
    }

    /// Default axis for a given [`ViewportId`] in this layout.
    ///
    /// Returns 0 (axial) for unrecognised combinations so callers always get a
    /// valid axis value.
    pub fn default_axis(&self, id: ViewportId) -> usize {
        match (self, id) {
            // 2×2
            (LayoutMode::TwoByTwo, ViewportId::TopLeft) => 0, // axial
            (LayoutMode::TwoByTwo, ViewportId::TopRight) => 1, // coronal
            (LayoutMode::TwoByTwo, ViewportId::BottomLeft) => 2, // sagittal
            (LayoutMode::TwoByTwo, ViewportId::BottomRight) => 0, // 3D uses axial data

            // 1+3
            (LayoutMode::OneMainThreeSmall, ViewportId::Main) => 0,
            (LayoutMode::OneMainThreeSmall, ViewportId::TopRight) => 0,
            (LayoutMode::OneMainThreeSmall, ViewportId::Center) => 1,
            (LayoutMode::OneMainThreeSmall, ViewportId::BottomRight) => 2,

            // Side-by-side: both axial by default (user may change)
            (LayoutMode::SideBySide, _) => 0,

            // Three horizontal: axial | coronal | sagittal
            (LayoutMode::ThreeHorizontal, ViewportId::Left) => 0,
            (LayoutMode::ThreeHorizontal, ViewportId::Center) => 1,
            (LayoutMode::ThreeHorizontal, ViewportId::Right) => 2,

            // Single / fallback
            _ => 0,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
            let json = serde_json::to_string(&mode).unwrap();
            let recovered: LayoutMode = serde_json::from_str(&json).unwrap();
            assert_eq!(
                mode, recovered,
                "{:?} serde round-trip must preserve value",
                mode
            );
        }
    }
}
