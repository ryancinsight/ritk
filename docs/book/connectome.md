# Connectome Construction and Graph Measures

A *connectome* is the graph whose nodes are anatomical regions and whose edges
summarise the streamlines running between them. `ritk-connectome` turns a
tractogram plus a [`Parcellation`](parcellation.md) into that graph, and then
measures it.

```text
Parcellation (labels) ──┐
                        ├──► ConnectivityMatrix ──► graph measures
Streamlines (polylines) ┘
```

The crate sits at the end of the diffusion-MRI pipeline documented in this part:
the [Gradient Schemes](diffusion_scheme.md) chapter validates acquisition
metadata, [Diffusion Models](ritk_diffusion.md) fits orientation models,
[Tractography](tractography.md) produces streamlines, the
[Parcellation](parcellation.md) chapter supplies the regions, and this chapter
builds the graph.

## What a connectome edge does and does not mean

An edge weight is a property of the *tractogram*, not of the brain. Streamline
counts are known not to be proportional to axonal density: tracking
systematically favours short, straight, high-anisotropy paths and systematically
loses long, curved, or crossing ones.

Two of the weightings below exist to remove the leading *geometric* parts of that
dependence — pathway length and region size — and none of them turns a count into
a measurement of connection strength. Comparison across subjects processed
identically is the defensible use; absolute interpretation of a single weight is
not.

## Construction

Two decisions separate a tractogram from a connectome, and both change the
answer.

```rust,ignore
use ritk_connectome::{
    ConnectomeConfig, EdgeWeighting, EndpointAssignment, build_connectivity_matrix,
};

let config = ConnectomeConfig::new()
    .with_assignment(EndpointAssignment::RadialSearch { radius_mm: 2.0 })
    .with_weighting(EdgeWeighting::InverseNodeVolume);

let matrix = build_connectivity_matrix(&parcellation, &streamlines, &config)?;
```

The command line and Python reach the same builder:

```bash
ritk tract connectome --tractogram tracks.tck --labels dseg.nii.gz \
                      --output matrix.json --measures measures.json
```

```python
matrix = ritk.connectome.build_connectivity_matrix(
    parcellation, streamlines, assignment_radius=2.0
)
```

### 1. Which region does an endpoint belong to?

| Assignment | Behaviour |
|---|---|
| `Terminal` | The label of the voxel the endpoint falls in |
| `RadialSearch { radius_mm }` | The nearest labelled voxel within the radius |

`Terminal` is exact, and exactly what discards streamlines terminating in white
matter: tracking stops at the grey/white boundary while a cortical parcellation
labels only grey matter, so rejection rates above half are ordinary. It is
correct when the parcellation covers the whole brain including white matter, or
when the streamlines were tracked to terminate inside grey matter by
construction.

`RadialSearch` searches the endpoint's own voxel first, so it can only *add*
assignments that `Terminal` would have dropped — it never moves an endpoint that
already sat inside a region. The radius trade-off is set out in the
[Parcellation](parcellation.md) chapter.

### 2. What does an edge weigh?

| Weighting | Units | What it is for |
|---|---|---|
| `StreamlineCount` | count | The raw, unnormalised connectome |
| `InverseLength` | mm⁻¹ | Divides out the length dependence of tracking |
| `InverseNodeVolume` | mm⁻³ | Divides out the region-size dependence |
| `MeanLength` | mm | The pathway's geometry, not a count of it |

**Pathway length.** A streamline is reconstructed step by step, and the
probability of surviving to its endpoint falls with the number of steps.
Long-range connections are therefore systematically under-counted relative to
short ones, by roughly the pathway length. `InverseLength` divides it back out.

**Region size.** A large region presents a large surface for streamlines to
terminate on, so its edges are heavier for reasons of geometry rather than
anatomy. This confounds comparison between regions of different size and between
subjects whose regions differ in size. `InverseNodeVolume` divides by the summed
node volumes.

**Mean length** is not a connectivity measure at all: an edge of weight 60 means
the streamlines joining those regions averaged 60 mm, whatever their number.
Useful paired with a count matrix, not as a substitute.

### Accounting

Every streamline lands in exactly one bucket, and the buckets sum to the whole:

| Field | Meaning |
|---|---|
| `total` | Streamlines supplied |
| `assigned` | Produced an inter-region edge |
| `intra_region` | Both endpoints resolved to the same region |
| `unassigned` | At least one endpoint no region could be found for |

The three outcomes partition the input exactly:
`assigned + intra_region + unassigned = total`. There is no bucket for a
malformed streamline, because a `Polyline` cannot hold fewer than two finite
points — its constructor refuses to build one.

This is kept alongside the matrix because a connectome is not interpretable
without it. A matrix built from a tractogram of which four fifths were discarded
is a different claim from one built from a tractogram of which a twentieth were,
and nothing in the weights distinguishes the two.

```rust,ignore
let accounting = matrix.accounting();
println!("{:.1}% of streamlines produced an edge", 100.0 * accounting.assigned_fraction());
```

## The connectivity matrix

`ConnectivityMatrix` stores a dense, **fully symmetric** \\(n \\times n\\) weight
matrix: both \\((i,j)\\) and \\((j,i)\\) carry the weight. A triangular layout
would halve the memory — for a whole-brain atlas a few hundred kilobytes either
way — at the cost of an index-ordering branch inside every measure that walks a
row, and the graph algorithms walk rows constantly.

Self-connections sit on the diagonal. They are recorded, because a tractogram's
intra-region streamlines are real, but excluded from degree, density, and every
path-based measure, where a self-loop is not a connection between two nodes.

| Method | Returns |
|---|---|
| `weight(source, target)` | Edge weight by label, `None` for an unknown label |
| `weight_at(i, j)` / `row(i)` | By matrix index |
| `edges()` | Every nonzero edge, self-connections included |
| `edge_count()` | Distinct connected pairs, self-connections excluded |
| `degree(label)` / `strength(label)` | Neighbour count / summed incident weight |
| `density()` | Connected pairs over possible pairs |
| `measures()` | Every graph measure below |

## Graph measures

```rust,ignore
let measures = matrix.measures();

println!("global efficiency  {:.3}", measures.global_efficiency());
println!("mean clustering    {:.3}", GraphMeasures::mean(measures.clustering()));
println!("communities        {}",    measures.communities().count());
println!("modularity         {:.3}", measures.communities().modularity());
```

They are computed together rather than one at a time because they share the
all-pairs shortest-path solution, which dominates the cost.

### The one convention everything rests on

A connectome edge weight is a measure of *connection strength*: a heavier edge
means the regions are more closely linked. Every shortest-path measure needs the
opposite — a *distance*, where a stronger link is a shorter step. The two are
related by inversion,

\\[ \\ell(i, j) = 1 / w(i, j) \\]

and the crate applies that inversion at the single point where the graph is
converted for path finding. Getting it backwards is the classic error in weighted
network analysis, and it does not fail loudly: it produces a path length that is
largest exactly where the connection is strongest, so every derived measure
inverts its meaning while remaining a plausible number.

### Segregation

| Measure | Definition |
|---|---|
| `clustering()` | \\(C = 2t / (k(k-1))\\) — the fraction of a node's possible neighbour links realised |
| `weighted_clustering()` | Onnela: triangles counted by the geometric mean of their three normalised weights |
| `local_efficiency()` | Global efficiency of the subgraph induced on a node's neighbours, with the node removed |
| `communities()` | Louvain partition, with its weighted modularity |

The binary clustering coefficient throws away everything the weights say: a
triangle closed by three heavy edges counts the same as one closed by three
negligible ones. The Onnela form replaces the triangle count with the sum of the
geometric means of the three normalised weights. The geometric mean is the choice
that makes the measure behave — it is zero if any of the three edges is absent,
so a missing edge closes no triangle, and it reduces exactly to the binary
coefficient when every present edge carries the maximum weight.

Community detection maximises modularity, the excess of within-community weight
over a strength-preserving null model:

\\[ Q = \\frac{1}{2m} \\sum_{ij} \\left[ w_{ij} - \\frac{k_i k_j}{2m} \\right] \\delta(c_i, c_j) \\]

The implementation visits nodes in **index order** rather than at random. The
published Louvain algorithm randomises, which makes the result vary between runs
on the same input — unacceptable for a measure that has to be compared between
subjects or reproduced from a paper. The cost is a partition that may differ from
the best one a randomised search would find, and the modularity reported is
always the modularity of the partition actually returned.

Modularity maximisation cannot resolve communities smaller than roughly
\\(\\sqrt{2m}\\) in total weight; below that, merging two genuinely separate
communities *increases* \\(Q\\). This is a property of the objective, not of the
search, so no amount of optimisation effort avoids it.

### Integration

| Measure | Definition |
|---|---|
| `characteristic_path_length()` | Mean shortest-path length over reachable pairs |
| `reachable_pair_fraction()` | Fraction of ordered pairs with a path |
| `global_efficiency()` | Mean *reciprocal* shortest-path length |
| `component_sizes()` | Connected component sizes, descending |

A real connectome is often not connected — an isolated region, or a parcellation
whose regions the tractogram never reached. That makes the characteristic path
length infinite, since some pair has no path. Two responses are in use and both
are wrong in one direction: averaging over only the reachable pairs understates
the cost of disconnection, and dropping the measure loses it entirely.

Global efficiency avoids the problem by averaging the reciprocal distance, where
an unreachable pair contributes exactly zero. It is therefore reported
unconditionally, and is the better-behaved summary.
`characteristic_path_length()` is reported over reachable pairs only, and must be
read together with `reachable_pair_fraction()`: a graph in fragments can show a
*short* characteristic path precisely because the long paths do not exist.

### Centrality and hubs

| Measure | Definition |
|---|---|
| `betweenness()` | Fraction of shortest paths passing through each node, normalised to \\([0,1]\\) |
| `rich_club()` | \\(\\Phi(k) = 2E_{>k} / (N_{>k}(N_{>k}-1))\\) at each degree threshold |

Betweenness identifies hubs whose *position* matters rather than whose degree is
large: a node of modest degree bridging two otherwise separate modules carries
enormous traffic, and degree does not see it. It is computed by Brandes'
algorithm, which never enumerates a path.

The rich-club coefficient asks whether the high-degree nodes are preferentially
wired to each other. \\(\\Phi(k)\\) rises with \\(k\\) in *any* graph, because
high-degree nodes have more edges and so are likelier to be connected by chance
alone — so **a rising raw curve is not by itself evidence of a rich club.** The
measure is the ratio against a null model that keeps every degree and rewires
everything else:

\\[ \\Phi_{\\text{norm}}(k) = \\Phi(k) / \\langle \\Phi_{\\text{random}}(k) \\rangle \\]

`normalised_rich_club` computes it. The ensemble is built by repeated
double-edge swaps, which preserve every node's degree exactly — so the club
membership at each threshold is identical in every sample, and only the edges
*among* the club change. That is what makes the ratio a statement about wiring
rather than about degree.

```rust,ignore
use ritk_connectome::measures::rich_club::{RandomisationConfig, normalised_rich_club};

let (levels, report) = normalised_rich_club(&matrix, RandomisationConfig::new(1000, 42))?;
for level in &levels {
    if let Some(ratio) = level.ratio {
        println!(
            "k={} ratio {ratio:.2} (random spread {:.2})",
            level.observed.degree, level.random_deviation
        );
    }
}
println!("rewiring acceptance {:.1}%", 100.0 * report.acceptance());
```

Ensemble size, swaps per edge, and the seed are explicit because each is a
study-design choice with no defensible default. The seed is fixed rather than
drawn, so a reported ratio can be reproduced.

**Read the ratio with the acceptance fraction.** A ratio near one means either
that the wiring is unremarkable *or* that the degree sequence never allowed it to
be otherwise — four nodes of degree five among eight degree-one leaves has
exactly six slots for edges between them, every pair, so every graph with that
sequence has a complete club and one is the true answer. A low acceptance
fraction means the graph was too constrained to rewire and the ensemble never
left where it started.


Each level also reports `mean_weight`, the weighted companion: a club can be
topologically complete while its edges are individually weak.

## Persistence

```rust,ignore
let json = matrix.to_json()?;
let restored = ConnectivityMatrix::from_json(&json)?;
```

## References

- Rubinov, M. & Sporns, O. (2010). Complex network measures of brain
  connectivity: Uses and interpretations. *NeuroImage* 52(3):1059–1069.
- Jones, D. K., Knösche, T. R. & Turner, R. (2013). White matter integrity, fiber
  count, and other fallacies. *NeuroImage* 73:239–254.
- Blondel, V. D. *et al.* (2008). Fast unfolding of communities in large
  networks. *J. Stat. Mech.* 2008:P10008.
- Brandes, U. (2001). A faster algorithm for betweenness centrality.
  *J. Math. Sociol.* 25(2):163–177.
- Onnela, J.-P. *et al.* (2005). Intensity and coherence of motifs in weighted
  complex networks. *Phys. Rev. E* 71:065103.
- Colizza, V. *et al.* (2006). Detecting rich-club ordering in complex networks.
  *Nature Physics* 2:110–115.
