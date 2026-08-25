"""Python surface for parcellations, connectomes, and graph measures.

The construction and every measure live in Rust and are tested there. What
these cover is the boundary: that NumPy arrays convert into the crates' types
with the axis order intact, that the results come back shaped as documented,
and that a bad argument raises rather than producing a plausible wrong answer.
"""

import numpy as np
import pytest

import ritk
from ritk import connectome


def strip_parcellation(spacing=(1.0, 1.0, 1.0)):
    """An 8-voxel strip with a parcel at each end and background between.

    Laid out along the *first* array axis, which is the slowest-varying index
    and therefore physical x under an identity direction. Putting it on the last
    axis instead would run it along z, and the streamlines below would cross the
    strip rather than travel through it.
    """
    labels = np.zeros((8, 1, 1), dtype=np.uint32)
    labels[0, 0, 0] = 1
    labels[1, 0, 0] = 1
    labels[6, 0, 0] = 2
    labels[7, 0, 0] = 2
    return connectome.Parcellation(
        labels,
        spacing=spacing,
        origin=(0.0, 0.0, 0.0),
        region_names=[(1, "Left"), (2, "Right")],
    )


def line(start, end):
    return np.array([start, end], dtype=np.float64)


def spanning():
    return line((0.0, 0.0, 0.0), (7.0, 0.0, 0.0))


# -- Parcellation ---------------------------------------------------------


def test_parcellation_reports_its_regions():
    parcellation = strip_parcellation()

    assert len(parcellation) == 8
    assert parcellation.region_count == 2
    assert parcellation.region_labels == [1, 2]
    assert parcellation.shape == [8, 1, 1]
    assert parcellation.name_of(2) == "Right"
    assert parcellation.name_of(99) is None


def test_labels_round_trip_through_the_array_boundary():
    parcellation = strip_parcellation()
    labels = parcellation.labels()

    assert labels.shape == (8, 1, 1)
    assert labels.dtype == np.uint32
    assert labels[0, 0, 0] == 1
    assert labels[3, 0, 0] == 0
    assert labels[7, 0, 0] == 2


def test_label_at_distinguishes_outside_from_unassigned():
    """Outside the field of view and unassigned within it are different claims."""
    parcellation = strip_parcellation()

    assert parcellation.label_at((0.0, 0.0, 0.0)) == 1
    assert parcellation.label_at((7.0, 0.0, 0.0)) == 2
    assert parcellation.label_at((3.0, 0.0, 0.0)) == 0
    assert parcellation.label_at((500.0, 0.0, 0.0)) is None


def test_spacing_is_read_in_image_axis_order():
    """The first spacing entry belongs to the first array axis.

    Reversing the two would place every voxel somewhere it is not, while still
    answering every query — so the order is asserted through a position rather
    than trusted.
    """
    parcellation = strip_parcellation(spacing=(4.0, 1.0, 1.0))

    # Voxel index 1 along the first axis sits at x = 4 mm, not 1 mm.
    assert parcellation.label_at((4.0, 0.0, 0.0)) == 1
    assert parcellation.label_at((1.0, 0.0, 0.0)) == 1  # rounds back to voxel 0
    assert parcellation.label_at((28.0, 0.0, 0.0)) == 2


def test_region_statistics_reports_size_and_position():
    parcellation = strip_parcellation(spacing=(2.0, 1.0, 1.0))
    statistics = parcellation.region_statistics()

    assert [entry["label"] for entry in statistics] == [1, 2]
    first = statistics[0]
    assert first["voxel_count"] == 2
    # Two voxels of 2 x 1 x 1 mm.
    assert first["volume_mm3"] == pytest.approx(4.0)
    # Centres at x = 0 and x = 2 average to x = 1.
    assert first["centroid"][0] == pytest.approx(1.0)


def test_an_all_background_volume_is_rejected():
    labels = np.zeros((4, 1, 1), dtype=np.uint32)
    with pytest.raises(ValueError, match="no labelled regions"):
        connectome.Parcellation(labels, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))


def test_a_degenerate_geometry_is_rejected():
    labels = np.ones((4, 1, 1), dtype=np.uint32)
    with pytest.raises(ValueError):
        connectome.Parcellation(labels, spacing=(0.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))


# -- Connectome construction ----------------------------------------------


def test_a_spanning_streamline_connects_the_two_parcels():
    parcellation = strip_parcellation()
    matrix = connectome.build_connectivity_matrix(parcellation, [spanning()])

    assert len(matrix) == 2
    assert matrix.region_labels == [1, 2]
    assert matrix.edge_count == 1
    weights = matrix.weights()
    assert weights.shape == (2, 2)
    assert weights[0, 1] == pytest.approx(1.0)
    assert weights[1, 0] == pytest.approx(1.0)


def test_the_radius_decides_how_much_of_the_tractogram_survives():
    """A streamline stopping short of both parcels is dropped without a radius."""
    parcellation = strip_parcellation()
    short = line((2.0, 0.0, 0.0), (5.0, 0.0, 0.0))

    dropped = connectome.build_connectivity_matrix(
        parcellation, [short], assignment_radius=0.0
    )
    assert dropped.accounting["unassigned"] == 1
    assert dropped.edge_count == 0

    recovered = connectome.build_connectivity_matrix(
        parcellation, [short], assignment_radius=2.0
    )
    assert recovered.accounting["assigned"] == 1
    assert recovered.edge_count == 1


def test_the_accounting_partitions_the_tractogram():
    parcellation = strip_parcellation()
    streamlines = [
        spanning(),
        spanning(),
        line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),  # stays inside parcel 1
        line((0.0, 0.0, 0.0), (500.0, 0.0, 0.0)),  # leaves the volume
    ]
    matrix = connectome.build_connectivity_matrix(
        parcellation, streamlines, assignment_radius=0.0
    )

    accounting = matrix.accounting
    assert accounting["total"] == 4
    assert (
        accounting["assigned"] + accounting["intra_region"] + accounting["unassigned"]
        == accounting["total"]
    )
    assert accounting["assigned_fraction"] == pytest.approx(0.5)


def test_every_weighting_is_accepted_and_changes_the_weight():
    parcellation = strip_parcellation()
    weights = {}
    for name in ("count", "inverse_length", "inverse_node_volume", "mean_length"):
        matrix = connectome.build_connectivity_matrix(
            parcellation, [spanning()], weighting=name
        )
        weights[name] = matrix.weights()[0, 1]

    assert weights["count"] == pytest.approx(1.0)
    assert weights["inverse_length"] == pytest.approx(1.0 / 7.0)
    assert weights["mean_length"] == pytest.approx(7.0)
    # Two parcels of two 1 mm cubes each: 4 mm3 summed.
    assert weights["inverse_node_volume"] == pytest.approx(0.25)


def test_an_unknown_weighting_is_rejected():
    parcellation = strip_parcellation()
    with pytest.raises(ValueError, match="unknown weighting"):
        connectome.build_connectivity_matrix(
            parcellation, [spanning()], weighting="sift2"
        )


def test_a_negative_radius_is_rejected():
    parcellation = strip_parcellation()
    with pytest.raises(ValueError, match="nonnegative"):
        connectome.build_connectivity_matrix(
            parcellation, [spanning()], assignment_radius=-1.0
        )


def test_a_malformed_streamline_array_is_rejected():
    parcellation = strip_parcellation()
    with pytest.raises(ValueError, match=r"\[N, 3\]"):
        connectome.build_connectivity_matrix(
            parcellation, [np.zeros((4, 2), dtype=np.float64)]
        )


def test_a_single_point_streamline_is_rejected():
    parcellation = strip_parcellation()
    with pytest.raises(ValueError):
        connectome.build_connectivity_matrix(
            parcellation, [np.zeros((1, 3), dtype=np.float64)]
        )


# -- Graph measures -------------------------------------------------------


def triangle_matrix():
    """Three parcels, each pair joined by one streamline."""
    labels = np.zeros((5, 1, 1), dtype=np.uint32)
    labels[0, 0, 0] = 1
    labels[2, 0, 0] = 2
    labels[4, 0, 0] = 3
    parcellation = connectome.Parcellation(
        labels, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
    )
    streamlines = [
        line((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        line((2.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
        line((0.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
    ]
    return connectome.build_connectivity_matrix(
        parcellation, streamlines, assignment_radius=0.0
    )


def test_measures_come_back_shaped_per_node():
    matrix = triangle_matrix()
    measures = matrix.measures()

    assert measures.node_count == 3
    assert measures.edge_count == 3
    assert measures.density == pytest.approx(1.0)
    for array in (
        measures.degree(),
        measures.strength(),
        measures.clustering(),
        measures.weighted_clustering(),
        measures.betweenness(),
        measures.local_efficiency(),
        measures.communities(),
    ):
        assert array.shape == (3,)


def test_a_complete_graph_has_the_values_its_shape_implies():
    measures = triangle_matrix().measures()

    # Every pair is directly joined, so no node is ever an intermediary and
    # every neighbourhood is closed.
    assert measures.betweenness() == pytest.approx(np.zeros(3))
    assert measures.clustering() == pytest.approx(np.ones(3))
    assert measures.global_efficiency == pytest.approx(1.0)
    assert measures.characteristic_path_length == pytest.approx(1.0)
    assert measures.reachable_pair_fraction == pytest.approx(1.0)
    assert measures.component_sizes == [3]
    # A complete graph has no sub-structure to partition.
    assert measures.community_count == 1


def test_rich_club_reports_a_ratio_and_its_acceptance():
    matrix = triangle_matrix()
    levels, acceptance = matrix.rich_club(ensemble_size=16, seed=7)

    assert 0.0 <= acceptance <= 1.0
    for level in levels:
        assert set(level) >= {
            "degree",
            "node_count",
            "coefficient",
            "random_mean",
            "random_deviation",
            "ratio",
        }


def test_rich_club_is_reproducible_from_its_seed():
    matrix = triangle_matrix()
    first, _ = matrix.rich_club(ensemble_size=16, seed=3)
    second, _ = matrix.rich_club(ensemble_size=16, seed=3)
    assert [entry["ratio"] for entry in first] == [entry["ratio"] for entry in second]


def test_an_empty_rich_club_ensemble_is_rejected():
    matrix = triangle_matrix()
    with pytest.raises(ValueError):
        matrix.rich_club(ensemble_size=0, seed=1)


# -- Module wiring --------------------------------------------------------


def test_the_submodule_is_importable_and_exported():
    import ritk.connectome as imported

    assert imported is connectome
    assert "connectome" in ritk.__all__
