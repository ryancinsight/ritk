"""Opt-in installed-wheel regression for the retained JPEG Lossless skull CT."""

from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pytest
from pydicom import dcmread
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, JPEGLosslessSV1

from ritk.io import anonymize_dicom_dir, read_image


SERIES_SIZES = (2, 34, 67, 89, 111)
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
def test_jpeg_lossless_skull_ct_volume_matches_native_reference(ct_series, variant):
    count, variants, reference = ct_series
    folder, by_size = variants[variant]
    uid, _ = by_size[count]
    actual = read_image(str(folder), series_instance_uid=uid)

    # Use the same reconstruction for independently decompressed samples. The
    # mixed-orientation acquisitions do not have one physical plane per file.
    assert actual.shape == reference.shape
    assert actual.origin == reference.origin
    assert actual.spacing == reference.spacing
    assert list(actual.direction) == list(reference.direction)
    np.testing.assert_array_equal(actual.to_numpy(), reference.to_numpy())
