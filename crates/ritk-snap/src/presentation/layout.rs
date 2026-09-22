//! Host-neutral pane roles and bounded responsive layout selection.

/// A RITK presentation pane role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PaneRole {
    /// One of the three orthogonal slice axes (`0` axial, `1` coronal,
    /// `2` sagittal).
    Axis(usize),
    /// A display-only scalar projection.
    Projection,
}

/// The number and arrangement of panes in a host surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PaneLayout {
    /// One full-size axial pane.
    Single,
    /// Two side-by-side orthogonal panes.
    Dual,
    /// A two-by-two orthogonal/projection grid.
    Quad,
}

/// A bounded pane rectangle in host pixels.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PaneRect {
    /// Horizontal origin.
    pub(crate) x: u32,
    /// Vertical origin.
    pub(crate) y: u32,
    /// Width excluding the separator gap.
    pub(crate) width: u32,
    /// Height excluding the separator gap.
    pub(crate) height: u32,
}

/// Minimum surface width for the dual-pane arrangement.
const DUAL_MIN_WIDTH: u32 = 640;
/// Minimum surface height for the dual-pane arrangement.
const DUAL_MIN_HEIGHT: u32 = 480;
/// Minimum surface width for the four-pane arrangement.
const QUAD_MIN_WIDTH: u32 = 960;
/// Minimum surface height for the four-pane arrangement.
const QUAD_MIN_HEIGHT: u32 = 640;

impl PaneLayout {
    /// Returns the stable DOM/CSS token for this arrangement.
    #[cfg(target_arch = "wasm32")]
    #[must_use]
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Dual => "dual",
            Self::Quad => "quad",
        }
    }
    /// Select a layout from a bounded host extent.
    #[must_use]
    pub(crate) const fn responsive(width: u32, height: u32) -> Self {
        if width >= QUAD_MIN_WIDTH && height >= QUAD_MIN_HEIGHT {
            Self::Quad
        } else if width >= DUAL_MIN_WIDTH && height >= DUAL_MIN_HEIGHT {
            Self::Dual
        } else {
            Self::Single
        }
    }

    /// Return the number of visible panes.
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub(crate) const fn pane_count(self) -> usize {
        match self {
            Self::Single => 1,
            Self::Dual => 2,
            Self::Quad => 4,
        }
    }

    /// Return pane roles in stable display order.
    #[must_use]
    pub(crate) const fn roles(self) -> &'static [PaneRole] {
        const SINGLE: [PaneRole; 1] = [PaneRole::Axis(0)];
        const DUAL: [PaneRole; 2] = [PaneRole::Axis(0), PaneRole::Axis(1)];
        const QUAD: [PaneRole; 4] = [
            PaneRole::Axis(0),
            PaneRole::Axis(1),
            PaneRole::Axis(2),
            PaneRole::Projection,
        ];
        match self {
            Self::Single => &SINGLE,
            Self::Dual => &DUAL,
            Self::Quad => &QUAD,
        }
    }

    /// Partition a surface into non-overlapping pane rectangles.
    ///
    /// The returned slots follow [`Self::roles`]. Unused slots are `None` and
    /// therefore cannot accidentally receive input. The separator is applied
    /// only between neighboring rows or columns.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn partition(
        self,
        surface_width: u32,
        surface_height: u32,
        gap: u32,
    ) -> anyhow::Result<[Option<PaneRect>; 4]> {
        debug_assert_eq!(self.roles().len(), self.pane_count());
        if surface_width == 0 || surface_height == 0 {
            anyhow::bail!("responsive pane surface dimensions must be nonzero");
        }
        let mut panes = [None; 4];
        match self {
            Self::Single => {
                panes[0] = Some(PaneRect {
                    x: 0,
                    y: 0,
                    width: surface_width,
                    height: surface_height,
                });
            }
            Self::Dual => {
                let available = surface_width
                    .checked_sub(gap)
                    .ok_or_else(|| anyhow::anyhow!("dual pane surface is narrower than its gap"))?;
                if available < 2 {
                    anyhow::bail!("dual pane surface cannot allocate two columns");
                }
                let left = available / 2 + available % 2;
                let right = available / 2;
                panes[0] = Some(PaneRect {
                    x: 0,
                    y: 0,
                    width: left,
                    height: surface_height,
                });
                panes[1] = Some(PaneRect {
                    x: left + gap,
                    y: 0,
                    width: right,
                    height: surface_height,
                });
            }
            Self::Quad => {
                let available_width = surface_width
                    .checked_sub(gap)
                    .ok_or_else(|| anyhow::anyhow!("quad pane surface is narrower than its gap"))?;
                let available_height = surface_height
                    .checked_sub(gap)
                    .ok_or_else(|| anyhow::anyhow!("quad pane surface is shorter than its gap"))?;
                if available_width < 2 || available_height < 2 {
                    anyhow::bail!("quad pane surface cannot allocate four panes");
                }
                let left = available_width / 2 + available_width % 2;
                let right = available_width / 2;
                let top = available_height / 2 + available_height % 2;
                let bottom = available_height / 2;
                panes[0] = Some(PaneRect {
                    x: 0,
                    y: 0,
                    width: left,
                    height: top,
                });
                panes[1] = Some(PaneRect {
                    x: left + gap,
                    y: 0,
                    width: right,
                    height: top,
                });
                panes[2] = Some(PaneRect {
                    x: 0,
                    y: top + gap,
                    width: left,
                    height: bottom,
                });
                panes[3] = Some(PaneRect {
                    x: left + gap,
                    y: top + gap,
                    width: right,
                    height: bottom,
                });
            }
        }
        Ok(panes)
    }
}

impl PaneRole {
    /// Returns the stable DOM token for this role.
    #[cfg(target_arch = "wasm32")]
    #[must_use]
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Axis(0) => "axial",
            Self::Axis(1) => "coronal",
            Self::Axis(2) => "sagittal",
            Self::Axis(_) => "axis",
            Self::Projection => "projection",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PaneLayout, PaneRole};

    #[test]
    fn responsive_selection_is_stable_at_each_boundary() {
        assert_eq!(PaneLayout::responsive(0, 0), PaneLayout::Single);
        assert_eq!(PaneLayout::responsive(639, 800), PaneLayout::Single);
        assert_eq!(PaneLayout::responsive(640, 480), PaneLayout::Dual);
        assert_eq!(PaneLayout::responsive(959, 640), PaneLayout::Dual);
        assert_eq!(PaneLayout::responsive(960, 640), PaneLayout::Quad);
    }

    #[test]
    fn roles_and_counts_are_value_semantic() {
        assert_eq!(PaneLayout::Single.pane_count(), 1);
        assert_eq!(PaneLayout::Dual.pane_count(), 2);
        assert_eq!(PaneLayout::Quad.pane_count(), 4);
        assert_eq!(PaneLayout::Quad.roles()[3], PaneRole::Projection);
    }

    #[test]
    fn partition_forms_a_disjoint_union() {
        for layout in [PaneLayout::Single, PaneLayout::Dual, PaneLayout::Quad] {
            let panes = layout.partition(101, 77, 3).expect("partition");
            let active = panes.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(active.len(), layout.pane_count());
            for (index, left) in active.iter().enumerate() {
                for right in active.iter().skip(index + 1) {
                    let separated = left.x + left.width <= right.x
                        || right.x + right.width <= left.x
                        || left.y + left.height <= right.y
                        || right.y + right.height <= left.y;
                    assert!(separated, "pane rectangles overlap: {left:?} and {right:?}");
                }
            }
            let max_x = active
                .iter()
                .map(|pane| pane.x + pane.width)
                .max()
                .expect("active panes");
            let max_y = active
                .iter()
                .map(|pane| pane.y + pane.height)
                .max()
                .expect("active panes");
            assert_eq!(max_x, 101);
            assert_eq!(max_y, 77);
        }
    }
}
