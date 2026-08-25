"""``ritk.registration.parcellate_with_atlases`` — producing a parcellation.

``ritk.connectome`` could consume a parcellation but nothing in the package
produced one: a caller had to bring their own label volume. These tests cover
the seam between the binding and ``ritk-registration``'s pipeline — that images
in become atlases, that the labels land on the subject's grid in the array
order Python expects, and that a disagreement is resolved by the named fusion
rule and reported in the agreement map.

The fixtures are deliberately anisotropic in both shape and spacing. A cubic
volume on an isotropic grid cannot fail an axis-order test: reverse the axes and
every array is the same length and every voxel the same size, so the defect
cancels itself and the test passes while the geometry is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

import ritk

# ``[Z, Y, X]``, unequal on every axis.
SHAPE = (10, 12, 14)
# Voxel size in mm, outermost axis first — unequal for the same reason.
SPACING = (2.0, 1.5, 1.0)
# Half-open index range of the foreground block per axis, offset differently on
# each so that a permutation moves it.
BLOCK = (slice(2, 5), slice(4, 8), slice(7, 12))
# Any value clearly above the background; the registration metric is
# scale-invariant.
FOREGROUND = 100.0

# A short schedule keeps the test quick. It is honest here because each atlas
# already sits on the subject, so there is nothing for more iterations to close.
ITERATIONS = [4, 2]


def block(value: float) -> np.ndarray:
    """A block of ``value`` on a background of zero."""
    volume = np.zeros(SHAPE, dtype=np.float32)
    volume[BLOCK] = value
    return volume


def image(volume: np.ndarray) -> ritk.Image:
    return ritk.Image(volume, spacing=SPACING)


@pytest.fixture
def subject() -> ritk.Image:
    return image(block(FOREGROUND))


def atlas(label: float) -> tuple[ritk.Image, ritk.Image]:
    """An atlas identical to the subject, labelling the block with ``label``.

    An atlas already on the subject is what makes the expected output exact
    rather than approximate: there is no deformation to recover, so the warp is
    the identity and the labels must arrive unchanged. Deforming it instead
    would measure the registration's accuracy, which its own tests cover.
    """
    return image(block(FOREGROUND)), image(block(label))


# -- The single-atlas path ----------------------------------------------------


def test_one_atlas_transfers_its_labels_onto_the_subject(subject):
    intensity, labels = atlas(7.0)

    result = ritk.registration.parcellate_with_atlases(
        subject, [intensity], [labels], iterations=ITERATIONS
    )

    parcellation = result.parcellation
    # PyO3 hands a fixed-size Rust array back as a list.
    assert parcellation.shape == list(SHAPE)
    assert parcellation.region_count == 1
    assert parcellation.region_labels == [7]

    expected = block(7.0).astype(np.uint32)
    np.testing.assert_array_equal(parcellation.labels(), expected)


def test_a_single_atlas_agrees_with_itself_everywhere(subject):
    intensity, labels = atlas(1.0)

    result = ritk.registration.parcellate_with_atlases(
        subject, [intensity], [labels], iterations=ITERATIONS
    )

    agreement = result.agreement
    assert agreement.shape == SHAPE
    # One atlas has nothing to disagree with; the map is a statement about the
    # method, not about the anatomy.
    np.testing.assert_allclose(agreement, 1.0, atol=1e-6)

    assert len(result.registration_quality) == 1
    assert np.isfinite(result.registration_quality[0])


# -- Fusion across several atlases --------------------------------------------


def test_the_majority_label_wins_and_the_dissent_is_recorded(subject):
    atlases = [atlas(1.0), atlas(1.0), atlas(2.0)]
    intensities = [pair[0] for pair in atlases]
    labels = [pair[1] for pair in atlases]

    result = ritk.registration.parcellate_with_atlases(
        subject, intensities, labels, fusion="majority", iterations=ITERATIONS
    )

    np.testing.assert_array_equal(
        result.parcellation.labels(), block(1.0).astype(np.uint32)
    )

    agreement = result.agreement
    # Two thirds is what distinguishes a real vote from a map that merely
    # reports whether a label was found — the failure a presence-only check
    # would pass.
    np.testing.assert_allclose(agreement[BLOCK], 2.0 / 3.0, atol=1e-5)
    # Outside the block every atlas said background, so the vote is unanimous.
    # That is what proves the two thirds above is a measured share and not a
    # constant written everywhere.
    assert agreement[0, 0, 0] == pytest.approx(1.0, abs=1e-6)

    assert len(result.registration_quality) == 3


def test_joint_fusion_agrees_when_the_atlases_are_interchangeable(subject):
    atlases = [atlas(1.0), atlas(1.0), atlas(2.0)]
    intensities = [pair[0] for pair in atlases]
    labels = [pair[1] for pair in atlases]

    result = ritk.registration.parcellate_with_atlases(
        subject, intensities, labels, fusion="joint", iterations=ITERATIONS
    )

    # Weighting by local match must reach the same answer when the atlases are
    # interchangeable; a different one would mean the weights read something
    # other than the local match.
    np.testing.assert_array_equal(
        result.parcellation.labels(), block(1.0).astype(np.uint32)
    )


def test_region_names_reach_the_parcellation(subject):
    intensity, labels = atlas(3.0)

    result = ritk.registration.parcellate_with_atlases(
        subject,
        [intensity],
        [labels],
        iterations=ITERATIONS,
        region_names=[(3, "precentral")],
    )

    assert result.parcellation.name_of(3) == "precentral"
    assert result.parcellation.name_of(4) is None


# -- The result feeds the connectome ------------------------------------------


def test_the_parcellation_is_accepted_by_the_connectome_builder(subject):
    """The point of producing a parcellation from Python is consuming it there.

    A result the connectome builder rejected would leave the pipeline exactly
    as broken as it was, so the join is asserted rather than assumed.
    """
    first, first_labels = atlas(1.0)
    # A second labelled region so there is an edge to find. The block spans
    # x indices 7..12 at 1 mm; splitting it gives two parcels a streamline can
    # run between.
    two_regions = block(1.0)
    two_regions[:, :, 10:12] = 2.0
    atlases_labels = [image(two_regions)]

    result = ritk.registration.parcellate_with_atlases(
        subject, [first], atlases_labels, iterations=ITERATIONS
    )
    parcellation = result.parcellation
    assert parcellation.region_count == 2

    # Physical positions of two voxels, one in each parcel, taken from the
    # parcellation itself so the streamline is in its frame by construction.
    statistics = {entry["label"]: entry for entry in parcellation.region_statistics()}
    start = np.asarray(statistics[1]["centroid"], dtype=np.float64)
    end = np.asarray(statistics[2]["centroid"], dtype=np.float64)
    streamline = np.stack([start, end])

    matrix = ritk.connectome.build_connectivity_matrix(
        parcellation, [streamline], assignment_radius=3.0
    )

    assert matrix.region_labels == [1, 2]
    weights = matrix.weights()
    assert weights.shape == (2, 2)
    assert weights[0, 1] == pytest.approx(1.0)
    assert weights[1, 0] == pytest.approx(weights[0, 1])


# -- Argument handling --------------------------------------------------------


def test_unpaired_atlases_are_rejected(subject):
    intensity, labels = atlas(1.0)
    other_intensity, _ = atlas(2.0)

    with pytest.raises(ValueError, match="2 intensities and 1 label volumes"):
        ritk.registration.parcellate_with_atlases(
            subject, [intensity, other_intensity], [labels]
        )


def test_no_atlas_is_rejected(subject):
    with pytest.raises(ValueError, match="at least one atlas"):
        ritk.registration.parcellate_with_atlases(subject, [], [])


def test_an_unknown_fusion_rule_names_the_alternatives(subject):
    intensity, labels = atlas(1.0)

    with pytest.raises(ValueError, match="majority"):
        ritk.registration.parcellate_with_atlases(
            subject, [intensity], [labels], fusion="mean", iterations=ITERATIONS
        )


def test_an_atlas_off_the_subject_grid_is_rejected_with_both_shapes(subject):
    """A registration recovers a deformation, never a resampling.

    A mismatched atlas silently accepted would produce labels for a brain of a
    different size, so the binding rejects it and names both shapes.
    """
    small = np.zeros((2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=r"lie on the subject's grid"):
        ritk.registration.parcellate_with_atlases(
            subject, [image(small)], [image(small)], iterations=ITERATIONS
        )


def test_an_empty_iteration_schedule_is_rejected(subject):
    intensity, labels = atlas(1.0)

    with pytest.raises(ValueError, match="at least one level"):
        ritk.registration.parcellate_with_atlases(
            subject, [intensity], [labels], iterations=[]
        )
