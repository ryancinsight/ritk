# Seeded Segmentation

Segmentation assigns a discrete anatomical or material label to each voxel.
RITK provides automatic thresholding, connected components, region growing,
clustering, watershed, level-set evolution, and ensemble consensus methods in
`ritk-segmentation`. This chapter starts with GrowCut because it makes the
interaction between image evidence and user-provided constraints explicit.

## GrowCut state and update

GrowCut is a synchronous cellular automaton. Voxel \(p\) carries an intensity
\(I_p\), label \(L_p\), and confidence \(C_p\in[0,1]\). Seed voxels start with
their supplied label and unit confidence; unseeded voxels start at label zero
and confidence zero. For each face-connected neighbor \(q\), RITK computes

\[
g(p,q)=1-\frac{|I_p-I_q|}{I_{\max}-I_{\min}},\qquad
A(q,p)=C_q\,g(p,q).
\]

If \(A(q,p)>C_p\), the neighbor wins the local competition and both state
variables update:

\[
(L_p,C_p)\leftarrow(L_q,A(q,p)).
\]

All voxels read the previous iteration and write the next one, so traversal
order does not change the result. The process stops only when the complete
\((L,C)\) state is stable or the configured iteration limit is reached. A
same-label confidence increase matters: it can strengthen a later attack at a
class boundary even though the current label did not change.

The [worked GrowCut figure](examples/growcut.md) makes this update visible over
time. Orange and cyan are competing labels, while the underlying light/dark
image is the evidence controlling attack strength. Pixels that retain that
grayscale image are still unlabeled. The figure evaluates \(g\) on each side
of its known boundary so the final circle follows from the update rule rather
than from the appearance of the last panel alone.

This follows the state and transition rule in Vezhnevets and Konouchine,
["GrowCut — Interactive Multi-Label N-D Image Segmentation by Cellular
Automata," §2.1](https://www.graphicon.ru/oldgr/en/publications/text/gc2005vk.pdf).
The paper establishes algorithm structure; RITK's executable tests and the
analytical example establish this implementation's behavior.

## Public native API

~~~rust,ignore
use coeus_core::SequentialBackend;
use ritk_segmentation::GrowCutFilter;

let backend = SequentialBackend;
let labels = GrowCutFilter::new(200)
    .apply_native(&image, &seeds, &backend)?;
~~~

`image` and `seeds` must have identical three-dimensional geometry. Positive
integer-valued seed voxels are immutable labels; zero denotes an unlabeled
voxel. `apply_native` preserves the input shape, origin, spacing, and
direction and reports shape or storage failures through `Result`.

## Choosing seeds

- Place at least one seed inside each class that the result must contain.
- Keep seeds away from uncertain boundaries.
- Add corrective seeds when different tissues have overlapping intensities.
- Treat the iteration limit as a resource bound, not as a convergence test.

The two-level phantom in the [worked GrowCut example](examples/growcut.md)
has an exact solution. Real CT and MR volumes do not: intensity overlap,
partial-volume voxels, artifacts, and seed placement all affect clinical
validity. Validate real segmentations against independent labels and report
overlap and boundary metrics.
