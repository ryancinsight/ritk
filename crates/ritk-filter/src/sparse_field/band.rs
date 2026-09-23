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
    propagate_layer(lists, phi, status, topo, num, cgv, 0, 1, 3, 1);
    propagate_layer(lists, phi, status, topo, num, cgv, 0, 2, 4, 2);
    for i in 1..(num - 2) {
        propagate_layer(
            lists,
            phi,
            status,
            topo,
            num,
            cgv,
            i,
            i + 2,
            i + 4,
            (i + 2) % 2,
        );
    }
}

/// `PropagateLayerValues` for one layer pair: value from the `frm` layer plus the
/// signed constant gradient, promoting voxels with no `frm` neighbour outward
/// (or out of the band entirely, once `promote` runs past the innermost layer).
#[allow(clippy::too_many_arguments)]
fn propagate_layer<T: SparseScalar>(
    lists: &mut SparseFieldLayers,
    phi: &mut [T],
    status: &mut [i32],
    topo: &GridTopology,
    num: i32,
    cgv: T,
    frm: i32,
    to: i32,
    promote: i32,
    inout: i32,
) {
    let offsets = topo.face_offsets();
    let delta = if inout == 1 { -cgv } else { cgv };
    let mut survivors: Vec<usize> = Vec::new();
    for f in lists.iter(to as usize).collect::<Vec<_>>() {
        if status[f] != to {
            continue;
        }
        let mut val = T::zero();
        let mut found = false;
        for &off in offsets {
            if let Some(g) = topo.neighbor(f, off) {
                if status[g] == frm {
                    let vt = phi[g];
                    if !found {
                        val = vt;
                    } else if inout == 1 {
                        val = if val > vt { val } else { vt };
                    } else {
                        val = if val < vt { val } else { vt };
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
