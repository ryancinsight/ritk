"""Opt-in installed-wheel regression for the retained JPEG Lossless skull CT."""

from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np
import pytest
from pydicom import dcmread
from pydicom.pixels import apply_modality_lut, pixel_array
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, JPEGLosslessSV1

from ritk.io import anonymize_dicom_dir, read_image


SERIES_SIZES = (2, 34, 67, 89, 111)
MIXED_SIZES = (2, 89, 111)
COHERENT_STACKS = (
    (89, 88, (0, 1, 0, 0, 0, -1)),
    (111, 110, (1, 0, 0, 0, 0, -1)),
)
DATA_ROOT = os.environ.get("RITK_DICOM_TEST_DATA")
pytestmark = pytest.mark.skipif(
    not DATA_ROOT, reason="set RITK_DICOM_TEST_DATA to the retained RITK test_data directory"
)


def _series_by_size(folder):
    by_uid = defaultdict(list)
    for path in sorted(folder.iterdir()):
        header = dcmread(path, stop_before_pixels=True)
        assert header.SOPClassUID == CTImageStorage
        assert header.file_meta.TransferSyntaxUID == JPEGLosslessSV1
        by_uid[str(header.SeriesInstanceUID)].append(path)
    assert sorted(map(len, by_uid.values())) == list(SERIES_SIZES)
    return {len(paths): (uid, paths) for uid, paths in by_uid.items()}


@pytest.fixture(scope="module")
def ct_study(tmp_path_factory):
    source = Path(DATA_ROOT) / "2_skull_ct" / "DICOM"
    by_variant = {"original": (source, _series_by_size(source))}
    temporary = tmp_path_factory.mktemp("skull-ct-import")
    for profile in ("basic", "enhanced"):
        export = temporary / profile
        assert anonymize_dicom_dir(str(source), str(export), profile=profile) == {
            "file_count": sum(SERIES_SIZES),
            "success_count": sum(SERIES_SIZES),
            "error_count": 0,
            "errors": [],
        }
        by_variant[profile] = (export, _series_by_size(export))
    return by_variant, temporary


@pytest.fixture(scope="module", params=SERIES_SIZES, ids=lambda count: f"{count}-instances")
def ct_series(request, ct_study):
    variants, temporary = ct_study
    count = request.param
    if count in MIXED_SIZES:
        return count, variants, None
    uid, paths = variants["original"][1][count]
    reference_folder = temporary / f"explicit-vr-{count}"
    reference_folder.mkdir()
    for index, path in enumerate(paths):
        dataset = dcmread(path)
        dataset.decompress(decoding_plugin="pylibjpeg", generate_instance_uid=False)
        assert dataset.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
        assert str(dataset.SeriesInstanceUID) == uid
        dataset.save_as(reference_folder / f"{index:06d}.dcm", enforce_file_format=True)
    return count, variants, read_image(str(reference_folder))


@pytest.mark.parametrize("variant", ("original", "basic", "enhanced"))
def test_jpeg_lossless_skull_ct_single_volume_contract(ct_series, variant):
    count, variants, reference = ct_series
    folder, by_size = variants[variant]
    uid, _ = by_size[count]
    if count in MIXED_SIZES:
        with pytest.raises(OSError, match="inconsistent ImageOrientationPatient"):
            read_image(str(folder), series_instance_uid=uid)
        return

    actual = read_image(str(folder), series_instance_uid=uid)
    assert actual.shape == reference.shape
    assert actual.origin == reference.origin
    assert actual.spacing == reference.spacing
    assert list(actual.direction) == list(reference.direction)
    np.testing.assert_array_equal(actual.to_numpy(), reference.to_numpy())


@pytest.mark.parametrize("variant", ("original", "basic", "enhanced"))
@pytest.mark.parametrize("series_count,depth,iop", COHERENT_STACKS)
def test_explicit_orientation_stack_matches_dicom_plane_geometry(
    ct_study, tmp_path, variant, series_count, depth, iop
):
    variants, _ = ct_study
    _, series = variants[variant]
    _, paths = series[series_count]
    instances = [(path, dcmread(path, stop_before_pixels=True)) for path in paths]
    orientations = Counter(tuple(ds.ImageOrientationPatient) for _, ds in instances)
    assert orientations.pop(iop) == depth
    assert list(orientations.values()) == [1]

    column_axis, row_axis = np.asarray(iop).reshape(2, 3)
    normal = np.cross(column_axis, row_axis)
    selected = sorted(
        ((path, ds) for path, ds in instances if tuple(ds.ImageOrientationPatient) == iop),
        key=lambda item: np.dot(item[1].ImagePositionPatient, normal),
    )
    positions = [np.dot(ds.ImagePositionPatient, normal) for _, ds in selected]
    assert np.all(np.diff(positions) > 0)
    assert np.median(np.diff(positions)) == 2.0

    for path, _ in selected:
        os.link(path, tmp_path / path.name)
    actual = read_image(str(tmp_path))
    first = selected[0][1]
    assert actual.shape == (depth, first.Rows, first.Columns)
    # A round trip through SI length units can round the DICOM decimal values.
    np.testing.assert_array_max_ulp(actual.spacing, [2.0, *first.PixelSpacing], maxulp=4)
    np.testing.assert_array_max_ulp(actual.origin, first.ImagePositionPatient[::-1], maxulp=4)
    np.testing.assert_array_equal(
        actual.direction, np.column_stack((column_axis, row_axis, normal)).ravel()
    )
    for plane, (path, _) in zip(actual.to_numpy(), selected, strict=True):
        ds = dcmread(path)
        expected = apply_modality_lut(pixel_array(ds, decoding_plugin="pylibjpeg"), ds)
        np.testing.assert_array_equal(plane, expected.astype(np.float32))
