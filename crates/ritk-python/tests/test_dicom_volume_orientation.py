"""Single-volume geometry contract exercised with tiny Part 10 files."""

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

from ritk.io import read_image


SERIES_UID = "2.25.101"
AXIAL = (1, 0, 0, 0, 1, 0)


def write_slice(path, index, orientation, samples):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = f"2.25.{104 + index}"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, preamble=bytes(128), file_meta=meta)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = "2.25.102"
    ds.SeriesInstanceUID = SERIES_UID
    ds.FrameOfReferenceUID = "2.25.103"
    ds.InstanceNumber = index + 1
    ds.Modality = "CT"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.ImageOrientationPatient = list(orientation)
    ds.ImagePositionPatient = [0, 0, 2 * index]
    ds.PixelSpacing = [1, 1]
    ds.SliceThickness = 2
    ds.Rows = ds.Columns = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = bytes(samples)
    ds.save_as(path, enforce_file_format=True)


@pytest.mark.parametrize("uid", (None, SERIES_UID), ids=("implicit", "explicit"))
@pytest.mark.parametrize(
    "orientation",
    [(1, 0, 0, 0, 0, -1), (-1, 0, 0, 0, -1, 0)],
    ids=("perpendicular", "same-normal-different-in-plane-axes"),
)
def test_incompatible_orientation_cannot_form_one_volume(tmp_path, uid, orientation):
    write_slice(tmp_path / "01.dcm", 0, AXIAL, [1, 2, 3, 4])
    write_slice(tmp_path / "02.dcm", 1, orientation, [5, 6, 7, 8])
    with pytest.raises(OSError, match="inconsistent ImageOrientationPatient"):
        read_image(str(tmp_path), series_instance_uid=uid)


def test_parallel_stack_keeps_samples_and_patient_geometry(tmp_path):
    write_slice(tmp_path / "z.dcm", 0, AXIAL, [1, 2, 3, 4])
    write_slice(tmp_path / "a.dcm", 1, AXIAL, [5, 6, 7, 8])
    image = read_image(str(tmp_path))
    assert image.shape == (2, 2, 2)
    assert image.spacing == (2.0, 1.0, 1.0)
    assert image.origin == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(image.direction, np.eye(3).ravel())
    np.testing.assert_array_equal(
        image.to_numpy(), [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    )
