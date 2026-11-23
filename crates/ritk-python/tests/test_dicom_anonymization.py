"""Exercise directory anonymization in the installed wheel on synthetic DICOM."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
from pydicom import dcmread, dcmwrite
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    BasicTextSRStorage,
    CTImageStorage,
    ExplicitVRLittleEndian,
    ParametricMapStorage,
    RLELossless,
    UID,
)

from ritk.io import anonymize_dicom_dir

UID_ROOT = "1.2.826.0.1.3680043.8.498.479"
STUDY_UID = f"{UID_ROOT}.1"
SERIES_UID = f"{UID_ROOT}.2"
FRAME_UID = f"{UID_ROOT}.3"
SOP_UIDS = (f"{UID_ROOT}.4.1", f"{UID_ROOT}.4.2")
PATIENT_NAME = "Synthetic^Filename"
PATIENT_ID = "SYNTHETIC-479"
PIXELS = b"\x03\x07\x12\x29"


def _write_instance(path, number, sop_uid, referenced_uid=None):
    meta = FileMetaDataset()
    meta.FileMetaInformationGroupLength = 0  # dcmwrite computes the encoded length.

    meta.FileMetaInformationVersion = b"\x00\x01"
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = f"{UID_ROOT}.5"
    ds = FileDataset(str(path), {}, preamble=bytes(128), file_meta=meta,
                     is_implicit_VR=False, is_little_endian=True)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = SERIES_UID
    ds.FrameOfReferenceUID = FRAME_UID
    ds.InstanceNumber = number
    ds.Modality = "CT"
    ds.PatientName = PATIENT_NAME
    ds.PatientID = PATIENT_ID
    ds.Rows = ds.Columns = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = PIXELS
    if referenced_uid is not None:
        reference = Dataset()
        reference.ReferencedSOPClassUID = CTImageStorage
        reference.ReferencedSOPInstanceUID = referenced_uid
        ds.ReferencedImageSequence = [reference]
    dcmwrite(path, ds)


@pytest.fixture
def source_study(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for index, uid in enumerate(SOP_UIDS):
        path = source / f"{PATIENT_NAME}-{uid}.dcm"
        _write_instance(path, index + 1, uid, SOP_UIDS[0] if index else None)
    (source / "notes.txt").write_bytes(b"not a DICOM file")
    nested = source / "nested"
    nested.mkdir()
    _write_instance(nested / f"{PATIENT_ID}.dcm", 3, f"{UID_ROOT}.4.3")
    return source


def _contents(directory):
    return {path.relative_to(directory): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def _export(source, output, **kwargs):
    return anonymize_dicom_dir(str(source), str(output), **kwargs)


def _success_count(count):
    return {"file_count": count, "success_count": count, "error_count": 0, "errors": []}


def test_numbered_dicom_export_preserves_uid_relationships_and_source(source_study, tmp_path):
    original = _contents(source_study)
    output = tmp_path / "new" / "export"

    stats = _export(source_study, output, profile="basic_replace_uids", uid_salt="filename-regression")

    assert stats == _success_count(2)
    exports = sorted(output.iterdir())
    assert [path.name for path in exports] == ["000001.dcm", "000002.dcm"]
    first, second = map(dcmread, exports)
    assert [first.InstanceNumber, second.InstanceNumber] == [1, 2]
    for attribute, original_uid in (("StudyInstanceUID", STUDY_UID),
                                    ("SeriesInstanceUID", SERIES_UID),
                                    ("FrameOfReferenceUID", FRAME_UID)):
        remapped = getattr(first, attribute)
        assert remapped == getattr(second, attribute)
        assert remapped != original_uid and UID(remapped).is_valid
    assert first.SOPInstanceUID != second.SOPInstanceUID
    assert second.ReferencedImageSequence[0].ReferencedSOPInstanceUID == first.SOPInstanceUID
    for ds in (first, second):
        assert ds.SOPInstanceUID not in SOP_UIDS and UID(ds.SOPInstanceUID).is_valid
        assert ds.SOPInstanceUID == ds.file_meta.MediaStorageSOPInstanceUID
        assert ds.SOPClassUID == ds.file_meta.MediaStorageSOPClassUID == CTImageStorage
        assert ds.PatientName == "ANONYMOUS"
        assert ds.PatientID == "ANON001"
        assert ds.PixelData == PIXELS
    assert _contents(source_study) == original


def test_numbered_dicom_export_skips_occupied_paths_and_keeps_earlier_exports(source_study, tmp_path):
    output = tmp_path / "export"
    output.mkdir()
    existing = output / "000001.dcm"
    existing.write_bytes(b"existing destination bytes")
    occupied_directory = output / "000002.dcm"
    occupied_directory.mkdir()
    (occupied_directory / "keep.txt").write_bytes(b"existing directory bytes")

    assert _export(source_study, output) == _success_count(2)
    before_retry = _contents(output)
    assert sorted(path.name for path in output.iterdir()) == [
        "000001.dcm", "000002.dcm", "000003.dcm", "000004.dcm",
    ]
    assert _export(source_study, output) == _success_count(2)

    assert existing.read_bytes() == b"existing destination bytes"
    assert sorted(path.name for path in output.iterdir()) == [
        "000001.dcm", "000002.dcm", "000003.dcm", "000004.dcm", "000005.dcm", "000006.dcm",
    ]
    for path, content in before_retry.items():
        assert (output / path).read_bytes() == content
    assert [dcmread(output / f"{number:06}.dcm").InstanceNumber for number in range(3, 7)] == [1, 2, 1, 2]


def test_concurrent_dicom_exports_do_not_overwrite_instances(source_study, tmp_path):
    output = tmp_path / "export"
    start = Barrier(2)

    def export(patient_id):
        start.wait(timeout=10)
        return _export(source_study, output, patient_id=patient_id)

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(export, ("COPY-A", "COPY-B"))) == [_success_count(2)] * 2

    assert sorted(path.name for path in output.iterdir()) == [f"{number:06}.dcm" for number in range(1, 5)]
    instances = [dcmread(path) for path in output.iterdir()]
    assert sorted((ds.PatientID, ds.InstanceNumber) for ds in instances) == [
        ("COPY-A", 1), ("COPY-A", 2), ("COPY-B", 1), ("COPY-B", 2),
    ]


def test_dicom_export_refuses_in_place_anonymization(source_study):
    original = _contents(source_study)

    aliases = [source_study / "nested" / ".."]
    case_alias = source_study.with_name("SOURCE")
    if case_alias.exists():
        aliases.append(case_alias)
    for alias in aliases:
        with pytest.raises(OSError, match="same directory"):
            _export(source_study, alias)

    assert _contents(source_study) == original


def test_dicom_export_reports_write_errors_without_publishing_partial_files(source_study, tmp_path):
    output = tmp_path / "export"
    original = _contents(source_study)

    # An Explicit VR Person Name field cannot encode a 65,536-byte value length.
    stats = _export(source_study, output, patient_name="X" * (1 << 16))

    assert stats["file_count"] == stats["error_count"] == len(stats["errors"]) == 2
    assert stats["success_count"] == 0
    assert sorted(path for path, _ in stats["errors"]) == sorted(
        str(path) for path in source_study.glob("*.dcm")
    )
    assert all(message for _, message in stats["errors"])
    assert list(output.iterdir()) == []
    assert _contents(source_study) == original


def test_header_actions_preserve_sequence_shape_and_value_representations(source_study, tmp_path):
    source = sorted(source_study.glob("*.dcm"))[0]
    original = dcmread(source)
    original["PatientName"].VR = "LO"  # A source sender used the wrong explicit VR.
    original["PatientName"].value = str(original["PatientName"].value)
    original.ContributingEquipmentSequence = [Dataset()]
    original.ContributingEquipmentSequence[0].Manufacturer = "Fixture scanner"
    dcmwrite(source, original)

    output = tmp_path / "output"
    assert _export(source_study, output, profile="enhanced") == _success_count(2)
    result = dcmread(output / "000001.dcm")
    assert result["PatientName"].VR == "PN"
    assert result["PatientID"].VR == "LO"
    assert list(result.ContributingEquipmentSequence[0].keys()) == [
        original.ContributingEquipmentSequence[0]["Manufacturer"].tag
    ]
    assert result.ContributingEquipmentSequence[0].Manufacturer == "Fixture scanner"


def test_enhanced_profile_protects_creation_and_contribution_metadata(source_study, tmp_path):
    source = sorted(source_study.glob("*.dcm"))[0]
    original = dcmread(source)
    original.InstanceCreationDate = "20240229"
    original.InstanceCreationTime = "123456"
    equipment = Dataset()
    equipment.Manufacturer = "Fixture scanner"
    equipment.ContributionDateTime = "20240229123456"
    equipment.ContributionDescription = "Fixture provenance"
    original.ContributingEquipmentSequence = [equipment]
    dcmwrite(source, original)

    output = tmp_path / "output"
    assert _export(source_study, output, profile="enhanced") == _success_count(2)
    result = dcmread(output / "000001.dcm")
    assert "InstanceCreationDate" not in result
    assert "InstanceCreationTime" not in result
    assert "ContributionDateTime" not in result.ContributingEquipmentSequence[0]
    assert "ContributionDescription" not in result.ContributingEquipmentSequence[0]
    assert result.ContributingEquipmentSequence[0].Manufacturer == "Fixture scanner"


def test_replaced_uids_use_the_uuid_arc_for_irradiation_events(source_study, tmp_path):
    event_uid = f"{UID_ROOT}.6"
    for source in source_study.glob("*.dcm"):
        original = dcmread(source)
        original.IrradiationEventUID = [event_uid, f"{UID_ROOT}.7"]
        dcmwrite(source, original)

    output = tmp_path / "output"
    assert _export(source_study, output, profile="enhanced") == _success_count(2)
    first, second = map(dcmread, sorted(output.iterdir()))
    assert first.IrradiationEventUID == second.IrradiationEventUID
    assert len(first.IrradiationEventUID) == len(set(first.IrradiationEventUID)) == 2
    for uid in (*first.IrradiationEventUID, first.SOPInstanceUID, second.SOPInstanceUID):
        assert UID(uid).is_valid and uid.startswith("2.25.")
        value = int(uid[5:])
        assert 0 < value < 1 << 128
        assert (value >> 76) & 0xF == 8
        assert (value >> 62) & 0x3 == 2


def test_flat_export_rejects_dicomdir_by_class_and_continues(source_study, tmp_path):
    from pydicom.uid import MediaStorageDirectoryStorage

    index = source_study / "000-index-renamed.dcm"
    original = dcmread(sorted(source_study.glob("*.dcm"))[0])
    meta = original.file_meta
    meta.MediaStorageSOPClassUID = MediaStorageDirectoryStorage
    directory = FileDataset(str(index), {}, file_meta=meta, preamble=bytes(128),
                            is_implicit_VR=False, is_little_endian=True)
    directory.FileSetID = "RITKTEST"
    directory.DirectoryRecordSequence = []
    dcmwrite(index, directory)

    output = tmp_path / "output"
    stats = _export(source_study, output)
    assert {key: stats[key] for key in ("file_count", "success_count", "error_count")} == {
        "file_count": 3, "success_count": 2, "error_count": 1
    }
    [(path, message)] = stats["errors"]
    assert path == str(index)
    assert "DICOMDIR" in message
    assert [entry.name for entry in sorted(output.iterdir())] == ["000001.dcm", "000002.dcm"]
    assert all(dcmread(path).SOPClassUID == CTImageStorage for path in output.iterdir())


def _write_rle_instance(path):
    _write_instance(path, 1, SOP_UIDS[0])
    ds = dcmread(path)
    ds.compress(RLELossless, encoding_plugin="pydicom", generate_instance_uid=False)
    dcmwrite(path, ds, enforce_file_format=True)
    compressed = dcmread(path)
    assert compressed.file_meta.TransferSyntaxUID == RLELossless
    assert compressed["PixelData"].is_undefined_length
    np.testing.assert_array_equal(compressed.pixel_array, [[3, 7], [18, 41]])
    return compressed


@pytest.mark.parametrize("cleaning", ["pixel", "all"])
def test_pixel_cleaning_rejects_encapsulated_input_without_publishing(cleaning, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    encapsulated = source / "1-encapsulated.dcm"
    _write_rle_instance(encapsulated)
    _write_instance(source / "2-native.dcm", 2, SOP_UIDS[1])
    input_before = _contents(source)
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "000001.dcm"
    existing.write_bytes(b"existing destination bytes")

    stats = _export(source, output, cleaning=cleaning)

    assert stats["file_count"] == 2 and stats["success_count"] == 1
    assert stats["error_count"] == 1
    [(path, detail)] = stats["errors"]
    assert path == str(encapsulated)
    assert "cannot clear encapsulated Pixel Data (7FE0,0010)" in detail
    assert sorted(path.name for path in output.iterdir()) == ["000001.dcm", "000002.dcm"]
    assert existing.read_bytes() == b"existing destination bytes"
    cleared = dcmread(output / "000002.dcm")
    assert cleared.SOPInstanceUID == SOP_UIDS[1]
    assert cleared.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
    assert cleared.PatientName == "ANONYMOUS"
    np.testing.assert_array_equal(cleared.pixel_array, [[0, 0], [0, 0]])
    assert _contents(source) == input_before


@pytest.mark.parametrize("profile", ["basic", "enhanced"])
def test_header_only_anonymization_preserves_encapsulated_pixels(profile, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = _write_rle_instance(source / "synthetic.dcm")
    output = tmp_path / "output"

    assert _export(source, output, profile=profile) == _success_count(1)

    exported = dcmread(output / "000001.dcm")
    assert exported.PatientName == "ANONYMOUS"
    assert exported.file_meta.TransferSyntaxUID == RLELossless
    assert exported.PixelData == original.PixelData
    np.testing.assert_array_equal(exported.pixel_array, [[3, 7], [18, 41]])


def test_explicit_pixel_cleaning_preserves_non_image_object_without_pixels(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    report = source / "report.dcm"
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = BasicTextSRStorage
    meta.MediaStorageSOPInstanceUID = SOP_UIDS[0]
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(report, {}, file_meta=meta, preamble=bytes(128))
    ds.SOPClassUID = BasicTextSRStorage
    ds.SOPInstanceUID = SOP_UIDS[0]
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = SERIES_UID
    ds.Modality = "SR"
    ds.PatientName = PATIENT_NAME
    ds.ValueType = "CONTAINER"
    ds.ContinuityOfContent = "SEPARATE"
    ds.CompletionFlag = "COMPLETE"
    ds.VerificationFlag = "UNVERIFIED"
    dcmwrite(report, ds, enforce_file_format=True)
    output = tmp_path / "output"

    assert _export(source, output, cleaning="all") == _success_count(1)

    exported = dcmread(output / "000001.dcm")
    assert exported.SOPClassUID == BasicTextSRStorage
    assert exported.PatientName == "ANONYMOUS"
    assert "PixelData" not in exported


@pytest.fixture(
    params=[
        ("FloatPixelData", "<f4", "7FE0,0008"),
        ("DoubleFloatPixelData", "<f8", "7FE0,0009"),
    ],
    ids=["float", "double-float"],
)
def floating_pixel_source(request, tmp_path):
    keyword, dtype, tag = request.param
    source = tmp_path / "source"
    source.mkdir()
    path = source / "synthetic.dcm"
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = ParametricMapStorage
    meta.MediaStorageSOPInstanceUID = SOP_UIDS[0]
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=bytes(128))
    ds.SOPClassUID = ParametricMapStorage
    ds.SOPInstanceUID = SOP_UIDS[0]
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = SERIES_UID
    ds.PatientName = PATIENT_NAME
    ds.Rows = ds.Columns = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.NumberOfFrames = 1
    expected = np.asarray([[1.25, -2.5], [0, 3.75]], dtype=dtype)
    ds.BitsAllocated = 8 * expected.dtype.itemsize
    setattr(ds, keyword, expected.tobytes())
    dcmwrite(path, ds, enforce_file_format=True)
    original = dcmread(path)
    assert "PixelData" not in original
    np.testing.assert_array_equal(original.pixel_array, expected)
    return path, keyword, tag, original, expected


@pytest.mark.parametrize("cleaning", ["pixel", "all"])
def test_explicit_pixel_cleaning_rejects_floating_point_data(
    cleaning, floating_pixel_source, tmp_path
):
    path, _, tag, _, _ = floating_pixel_source
    input_before = path.read_bytes()
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "000001.dcm"
    existing.write_bytes(b"existing destination bytes")

    stats = _export(path.parent, output, cleaning=cleaning)

    assert stats["file_count"] == stats["error_count"] == 1
    assert stats["success_count"] == 0
    [(error_path, detail)] = stats["errors"]
    assert error_path == str(path)
    assert f"cannot clear floating-point Pixel Data ({tag})" in detail
    assert list(output.iterdir()) == [existing]
    assert existing.read_bytes() == b"existing destination bytes"
    assert path.read_bytes() == input_before


def test_default_anonymization_preserves_floating_point_data(floating_pixel_source, tmp_path):
    path, keyword, _, original, expected = floating_pixel_source
    output = tmp_path / "output"

    assert _export(path.parent, output) == _success_count(1)

    exported = dcmread(output / "000001.dcm")
    assert exported.PatientName == "ANONYMOUS"
    assert exported.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
    assert getattr(exported, keyword) == getattr(original, keyword)
    np.testing.assert_array_equal(exported.pixel_array, expected)
