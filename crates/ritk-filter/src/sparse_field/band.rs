//! ITK's layer-value propagation and status-list cascade: the list bookkeeping
//! the `ApplyUpdate` loop performs once per iteration.

use std::collections::VecDeque;

use super::layers::{move_to, SparseFieldLayers};
use super::scalar::SparseScalar;
use super::topology::GridTopology;

/// Not in the band.
pub(crate) const ST_NULL: i32 = -1;
/// Marked `Changing` by the cascade (transiently outside the numbered layers).
pub(crate) const ST_CHG: i32 = -2;
/// Moved outward, waiting to be assigned a layer.
pub(crate) const ST_CUP: i32 = -3;
/// Moved inward, waiting to be assigned a layer.
pub(crate) const ST_CDN: i32 = -4;

#[derive(Clone, Copy)]
enum LevelSetSide {
    /// Negative signed-distance layers.
    Inside,
    /// Positive signed-distance layers.
    Outside,
}

/// The source, target, promotion layer and sign convention for one transition.
#[derive(Clone, Copy)]
struct LayerTransition {
    from: i32,
    to: i32,
    promote: i32,
    side: LevelSetSide,
}

/// `PropagateAllLayerValues`: layer 1 and 2 against the active layer, then every
/// deeper layer against the one two levels in.
pub(crate) fn propagate_all<T: SparseScalar>(
    lists: &mut SparseFieldLayers,
    phi: &mut [T],
    status: &mut [i32],
    topo: &GridTopology,
    num: i32,
    cgv: T,
) {
    propagate_layer(
        lists,
        phi,
        status,
        topo,
        num,
        cgv,
        LayerTransition {
            from: 0,
            to: 1,
            promote: 3,
            side: LevelSetSide::Inside,
        },
    );
    propagate_layer(
        lists,
        phi,
        status,
        topo,
        num,
        cgv,
        LayerTransition {
            from: 0,
            to: 2,
            promote: 4,
            side: LevelSetSide::Outside,
        },
    );
    for i in 1..(num - 2) {
        let to = i + 2;
        propagate_layer(
            lists,
            phi,
            status,
            topo,
            num,
            cgv,
            LayerTransition {
                from: i,
                to,
                promote: i + 4,
                side: if to % 2 == 1 {
                    LevelSetSide::Inside
                } else {
                    LevelSetSide::Outside
                },
            },
        );
    }
}

/// Propagate values from a source layer into its target layer, promoting voxels
/// without a source neighbour outward or removing them beyond the band.
fn propagate_layer<T: SparseScalar>(
    lists: &mut SparseFieldLayers,
    phi: &mut [T],
    status: &mut [i32],
    topo: &GridTopology,
    num: i32,
    cgv: T,
    transition: LayerTransition,
) {
    let LayerTransition {
        from,
        to,
        promote,
        side,
    } = transition;
    let offsets = topo.face_offsets();
    let delta = match side {
        LevelSetSide::Inside => -cgv,
        LevelSetSide::Outside => cgv,
    };
    let mut survivors: Vec<usize> = Vec::new();
    for f in lists.iter(to as usize).collect::<Vec<_>>() {
        if status[f] != to {
            continue;
        }
        let mut val = T::zero();
        let mut found = false;
        for &off in offsets {
            if let Some(g) = topo.neighbor(f, off) {
                if status[g] == from {
                    let vt = phi[g];
                    if !found {
                        val = vt;
                    } else {
                        val = match side {
                            LevelSetSide::Inside => {
                                if val > vt {
                                    val
                                } else {
                                    vt
                                }
                            }
                            LevelSetSide::Outside => {
                                if val < vt {
                                    val
                                } else {
                                    vt
                                }
                            }
                        };
                    }
                    found = true;
                }
            }
        }
        if found {
            phi[f] = val + delta;
            survivors.push(f);
        } else if promote > num - 1 {
            status[f] = ST_NULL;
        } else {
            move_to(lists, status, f, promote);
        }
    }
    lists.replace(to as usize, &survivors);
}

/// `ProcessStatusList`: consume `inl` from the front, move each voxel to `ct`,
/// and mark its `sr`-status neighbours as `Changing` into a fresh output list.
pub(crate) fn process_status_list(
    lists: &mut SparseFieldLayers,
    status: &mut [i32],
    topo: &GridTopology,
    mut inl: VecDeque<usize>,
    ct: i32,
    sr: i32,
) -> VecDeque<usize> {
    let mut outl: VecDeque<usize> = VecDeque::new();
    while let Some(f) = inl.pop_front() {
        move_to(lists, status, f, ct);
        for &off in topo.face_offsets() {
            if let Some(g) = topo.neighbor(f, off) {
                if status[g] == sr {
                    move_to(lists, status, g, ST_CHG);
                    outl.push_front(g);
                }
            }
        }
    }
    outl
}
