//! Watershed segment hierarchy used by isolated watershed.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use super::isolated::{neighbours, watershed_basins_gd};

#[derive(Clone, Copy, Debug)]
struct Merge {
    from: usize,
    to: usize,
    saliency: f32,
}

#[derive(Clone, Copy, Debug)]
struct Candidate(Merge);

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.saliency.total_cmp(&self.0.saliency)
    }
}

#[derive(Debug)]
struct Segment {
    minimum: f32,
    edges: Vec<Edge>,
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    label: usize,
    height: f32,
}

/// Initial watershed labels plus the dynamic ITK-style merge hierarchy.
pub(super) struct WatershedHierarchy {
    initial_labels: Box<[usize]>,
    merges: Box<[Merge]>,
    maximum_saliency: f32,
}

impl WatershedHierarchy {
    pub(super) fn build(
        gradient: &[f32],
        dimensions: [usize; 3],
        threshold: f32,
        upper_level: f64,
    ) -> Self {
        let (minimum, maximum) = gradient.iter().copied().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
        );
        let floor = threshold.mul_add(maximum - minimum, minimum);
        let clamped: Vec<f32> = gradient
            .iter()
            .copied()
            .map(|value| value.max(floor))
            .collect();
        let representatives = watershed_basins_gd(&clamped, dimensions);
        let mut label_by_representative = BTreeMap::new();
        for representative in representatives.iter().copied() {
            let next = label_by_representative.len();
            label_by_representative
                .entry(representative)
                .or_insert(next);
        }
        let initial_labels: Vec<usize> = representatives
            .iter()
            .map(|representative| label_by_representative[representative])
            .collect();
        let mut minima = vec![f32::INFINITY; label_by_representative.len()];
        let mut edge_maps = vec![BTreeMap::<usize, (f32, usize)>::new(); minima.len()];
        let mut edge_order = 0;
        for (index, (&label, &value)) in initial_labels.iter().zip(&clamped).enumerate() {
            minima[label] = minima[label].min(value);
            for neighbor in neighbours(index, dimensions) {
                let neighbor_label = initial_labels[neighbor];
                if neighbor_label == label {
                    continue;
                }
                let height = value.max(clamped[neighbor]);
                edge_maps[label]
                    .entry(neighbor_label)
                    .and_modify(|current| current.0 = current.0.min(height))
                    .or_insert_with(|| {
                        let edge = (height, edge_order);
                        edge_order += 1;
                        edge
                    });
            }
        }
        let mut segments: Vec<Segment> = minima
            .into_iter()
            .zip(edge_maps)
            .map(|(minimum, edges)| {
                let mut edges: Vec<_> = edges
                    .into_iter()
                    .map(|(label, (height, order))| (order, Edge { label, height }))
                    .collect();
                edges.sort_by_key(|(order, _)| *order);
                let mut edges: Vec<_> = edges.into_iter().map(|(_, edge)| edge).collect();
                edges.sort_by(|left, right| left.height.total_cmp(&right.height));
                Segment { minimum, edges }
            })
            .collect();
        let merges = generate_merges(&mut segments, upper_level * f64::from(maximum - minimum));
        let maximum_saliency = merges.last().map_or(0.0, |merge| merge.saliency).max(0.0);
        Self {
            initial_labels: initial_labels.into_boxed_slice(),
            merges: merges.into_boxed_slice(),
            maximum_saliency,
        }
    }

    pub(super) fn labels_at(&self, level: f64) -> Vec<usize> {
        let mut parent: Vec<usize> =
            (0..=self.initial_labels.iter().copied().max().unwrap_or(0)).collect();
        let merge_limit = level * f64::from(self.maximum_saliency);
        for merge in self.merges.iter().copied() {
            if f64::from(merge.saliency) > merge_limit {
                break;
            }
            let from = find(&mut parent, merge.from);
            let to = find(&mut parent, merge.to);
            if from != to {
                parent[from] = to;
            }
        }
        self.initial_labels
            .iter()
            .copied()
            .map(|label| find(&mut parent, label))
            .collect()
    }
}

fn generate_merges(segments: &mut [Segment], merge_limit: f64) -> Vec<Merge> {
    let mut parent: Vec<usize> = (0..segments.len()).collect();
    let mut heap = BinaryHeap::new();
    for from in 0..segments.len() {
        if let Some(candidate) = candidate_for(from, segments, &mut parent) {
            if f64::from(candidate.saliency) < merge_limit {
                heap.push(Candidate(candidate));
            }
        }
    }
    let mut merges = Vec::with_capacity(segments.len().saturating_sub(1));
    while let Some(Candidate(candidate)) = heap.pop() {
        if f64::from(candidate.saliency) > merge_limit {
            break;
        }
        let from = find(&mut parent, candidate.from);
        let to = find(&mut parent, candidate.to);
        if from != candidate.from || from == to {
            continue;
        }
        merges.push(Merge {
            from,
            to,
            saliency: candidate.saliency,
        });
        merge_segments(from, to, segments, &mut parent);
        if let Some(next) = candidate_for(to, segments, &mut parent) {
            heap.push(Candidate(next));
        }
    }
    merges
}

fn candidate_for(from: usize, segments: &mut [Segment], parent: &mut [usize]) -> Option<Merge> {
    let mut discarded = 0;
    for edge in &mut segments[from].edges {
        edge.label = find(parent, edge.label);
        if edge.label == from {
            discarded += 1;
        } else {
            break;
        }
    }
    if discarded > 0 {
        segments[from].edges.drain(..discarded);
    }
    segments[from].edges.first().map(|edge| Merge {
        from,
        to: edge.label,
        saliency: edge.height - segments[from].minimum,
    })
}

fn merge_segments(from: usize, to: usize, segments: &mut [Segment], parent: &mut [usize]) {
    segments[to].minimum = segments[to].minimum.min(segments[from].minimum);
    let from_edges = std::mem::take(&mut segments[from].edges);
    let to_edges = std::mem::take(&mut segments[to].edges);
    let mut merged = Vec::with_capacity(from_edges.len() + to_edges.len());
    let mut seen = BTreeSet::new();
    let (mut from_index, mut to_index) = (0, 0);
    while from_index < from_edges.len() && to_index < to_edges.len() {
        let mut from_edge = from_edges[from_index];
        let mut to_edge = to_edges[to_index];
        from_edge.label = find(parent, from_edge.label);
        to_edge.label = find(parent, to_edge.label);
        if seen.contains(&to_edge.label) || to_edge.label == from {
            to_index += 1;
            continue;
        }
        if seen.contains(&from_edge.label) || from_edge.label == to {
            from_index += 1;
            continue;
        }
        if from_edge.height < to_edge.height {
            seen.insert(from_edge.label);
            merged.push(from_edge);
            from_index += 1;
        } else {
            seen.insert(to_edge.label);
            merged.push(to_edge);
            to_index += 1;
        }
    }
    for mut edge in from_edges.into_iter().skip(from_index) {
        edge.label = find(parent, edge.label);
        if edge.label != to && seen.insert(edge.label) {
            merged.push(edge);
        }
    }
    for mut edge in to_edges.into_iter().skip(to_index) {
        edge.label = find(parent, edge.label);
        if edge.label != from && seen.insert(edge.label) {
            merged.push(edge);
        }
    }
    segments[to].edges = merged;
    parent[from] = to;
}

fn find(parent: &mut [usize], mut label: usize) -> usize {
    while parent[label] != label {
        parent[label] = parent[parent[label]];
        label = parent[label];
    }
    label
}

#[cfg(test)]
mod tests {
    use super::{merge_segments, Edge, Segment};

    #[test]
    fn equal_height_merge_retains_existing_destination_edge_first() {
        let mut segments = vec![
            Segment {
                minimum: 0.0,
                edges: vec![Edge {
                    label: 3,
                    height: 5.0,
                }],
            },
            Segment {
                minimum: 1.0,
                edges: vec![Edge {
                    label: 2,
                    height: 5.0,
                }],
            },
            Segment {
                minimum: 2.0,
                edges: Vec::new(),
            },
            Segment {
                minimum: 3.0,
                edges: Vec::new(),
            },
        ];
        let mut parent = vec![0, 1, 2, 3];
        merge_segments(0, 1, &mut segments, &mut parent);
        assert_eq!(
            segments[1]
                .edges
                .iter()
                .map(|edge| edge.label)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(parent, vec![1, 1, 2, 3]);
    }
}
