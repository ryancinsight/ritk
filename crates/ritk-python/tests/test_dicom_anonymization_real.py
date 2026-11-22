"""Opt-in installed-wheel acceptance on the documented, real DICOM fixtures."""

import hashlib
import os
import re
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from pydicom.multival import MultiValue
from pydicom.uid import CTImageStorage, MRImageStorage, UID

from ritk.io import anonymize_dicom_dir, read_image

DATA_ROOT = os.environ.get("RITK_DICOM_TEST_DATA")
pytestmark = pytest.mark.skipif(
    not DATA_ROOT, reason="set RITK_DICOM_TEST_DATA to the RITK test_data directory"
)
UID_ATTRIBUTES = (
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID",
    "IrradiationEventUID", "ReferencedSOPInstanceUID",
)
IMAGE_ATTRIBUTES = """
    Rows Columns SamplesPerPixel PhotometricInterpretation PlanarConfiguration
    BitsAllocated BitsStored HighBit PixelRepresentation NumberOfFrames
    ImageType Modality InstanceNumber InStackPositionNumber
    ImagePositionPatient ImageOrientationPatient PixelSpacing SliceLocation
    SliceThickness SpacingBetweenSlices PatientPosition GantryDetectorTilt
    PixelPaddingValue PixelPaddingRangeLimit SmallestImagePixelValue LargestImagePixelValue
    RescaleSlope RescaleIntercept RescaleType ModalityLUTSequence RealWorldValueMappingSequence
    WindowCenter WindowWidth VOILUTFunction VOILUTSequence
    LossyImageCompression LossyImageCompressionRatio LossyImageCompressionMethod
    EchoTime RepetitionTime FlipAngle MagneticFieldStrength ImagingFrequency
    PixelBandwidth AcquisitionMatrix KVP ConvolutionKernel
""".split()
TYPE_TWO_ATTRIBUTES = """
    PatientName PatientID PatientBirthDate PatientSex StudyDate StudyTime
    ReferringPhysicianName StudyID AccessionNumber SeriesNumber InstanceNumber SliceThickness
""".split()
EMPTY_TEMPORAL_ATTRIBUTES = (
    "StudyDate", "StudyTime", "SeriesDate", "SeriesTime", "AcquisitionDate",
    "AcquisitionTime", "ContentDate", "ContentTime",
)
REMOVED_PROVENANCE = {
    "InstanceCreationDate", "InstanceCreationTime", "ContributionDateTime",
    "ContributionDescription",
}
TEXT_VRS = {"AE", "CS", "LO", "LT", "PN", "SH", "ST", "UC", "UR", "UT"}


def _source_images(folder, sop_class):
    for path in sorted(folder.iterdir()):
        if path.is_file():
            try:
                header = dcmread(path, stop_before_pixels=True)
            except InvalidDicomError:
                continue
            assert header.SOPClassUID == sop_class
            yield path, header


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.digest()


def _manifest(folder):
    return {path.relative_to(folder): _sha256(path) for path in folder.rglob("*") if path.is_file()}


def _identifier_pattern(headers):
    identifiers = {
        str(header.get(key, "")).strip().casefold()
        for header in headers for key in ("PatientName", "PatientID")
        if str(header.get(key, "")).strip()
    }
    assert identifiers, "fixture has no populated patient identifiers to check"
    choices = "|".join(re.escape(value) for value in sorted(identifiers))
    return re.compile(r"(?<!\w)(?:" + choices + r")(?!\w)")


def _values(element):
    return element.value if isinstance(element.value, MultiValue) else (element.value,)


def _retained_fields(before, after):
    assert set(after.keys()) <= set(before.keys()), "export injected an absent attribute"
    for current in after:
        original = before[current.tag]
        yield original, current
        if current.VR == "SQ":
            assert len(original.value) == len(current.value), "changed sequence item count"
            for source_item, exported_item in zip(original.value, current.value):
                yield from _retained_fields(source_item, exported_item)


def _check_uid(source, output, profile, forward, reverse):
    assert UID(output).is_valid
    if profile == "basic":
        assert source == output
    else:
        assert source != output
        assert output.startswith("2.25.")
        encoded = int(output[5:])
        assert 0 < encoded < 1 << 128
        replacement = uuid.UUID(int=encoded)
        assert replacement.variant == uuid.RFC_4122 and replacement.version == 8
    assert forward.setdefault(source, output) == output
    assert reverse.setdefault(output, source) == source


def _check_headers(before, after, profile, identifiers, forward, reverse):
    assert after.file_meta.TransferSyntaxUID == before.file_meta.TransferSyntaxUID
    assert after.file_meta.MediaStorageSOPClassUID == after.SOPClassUID == before.SOPClassUID
    assert after.file_meta.MediaStorageSOPInstanceUID == after.SOPInstanceUID
    assert all(key in after and after.get(key) for key in UID_ATTRIBUTES[:4])
    assert all(key in after and after.get(key) is not None for key in (
        "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns"
    ))
    assert all(key in after for key in TYPE_TWO_ATTRIBUTES if key in before)
    assert after["PatientName"].VR == "PN" and str(after.PatientName) == "ANONYMOUS"
    assert after["PatientID"].VR == "LO" and after.PatientID == "ANON001"
    assert all(not after.get(key) for key in (
        "AccessionNumber", "PatientBirthDate", "PatientSex", "ReferringPhysicianName", "StudyID"
    ))
    assert all(after.get(key) == before.get(key) for key in IMAGE_ATTRIBUTES)
    if profile == "enhanced":
        assert all(not after.get(key) for key in EMPTY_TEMPORAL_ATTRIBUTES)
        assert "StudyDescription" not in after and "SeriesDescription" not in after
    else:
        assert all(after.get(key) == before.get(key) for key in EMPTY_TEMPORAL_ATTRIBUTES)
    source_private = sum(element.tag.is_private for element in before.iterall())
    after_private = sum(element.tag.is_private for element in after.iterall())
    assert after_private == (source_private if profile == "basic" else 0)

    seen = Counter()
    for original, current in _retained_fields(before, after):
        assert current.keyword not in REMOVED_PROVENANCE
        if current.VR in TEXT_VRS:
            assert all(not identifiers.search(str(value).casefold()) for value in _values(current)), \
                f"source identifier survived in {current.keyword}"
        if current.keyword in UID_ATTRIBUTES:
            av, bv = _values(original), _values(current)
            assert len(av) == len(bv)
            for source_uid, output_uid in zip(av, bv):
                _check_uid(source_uid, output_uid, profile, forward, reverse)
            seen[current.keyword] += len(av)
    return seen


@pytest.mark.parametrize(
    "dataset,count,series_sizes,sop_class,event_occurrences,internal_references",
    (
        ("2_head_mri_t2", 94, [94], MRImageStorage, 0, 0),
        ("2_skull_ct", 303, [2, 34, 67, 89, 111], CTImageStorage, 103, 101),
    ),
)
@pytest.mark.parametrize("profile", ("basic", "enhanced"))
def test_real_series_export(
    tmp_path, dataset, count, series_sizes, sop_class, event_occurrences, internal_references, profile
):
    from pydicom.pixels import apply_modality_lut, apply_voi_lut, pixel_array

    fixture = Path(DATA_ROOT) / dataset
    source = fixture / "DICOM"
    source_manifest = _manifest(fixture)
    inputs = list(_source_images(source, sop_class))
    assert len(inputs) == count
    identifiers = _identifier_pattern(header for _, header in inputs)
    source_sops = {header.SOPInstanceUID for _, header in inputs}
    assert len(source_sops) == count
    assert sum("IrradiationEventUID" in header for _, header in inputs) == event_occurrences

    output = tmp_path / "export"
    stats = anonymize_dicom_dir(str(source), str(output), profile=profile)
    assert stats == {"file_count": count, "success_count": count, "error_count": 0, "errors": []}
    outputs = sorted(output.iterdir())
    assert [path.name for path in outputs] == [f"{index:06d}.dcm" for index in range(1, count + 1)]
    assert not {path.name for path, _ in inputs}.intersection(path.name for path in outputs)
    assert not any(identifiers.search(path.name.casefold()) for path in outputs)

    forward, reverse = {}, {}
    output_series = Counter()
    output_studies, output_sops, output_frames = set(), set(), set()
    output_references = []
    output_uids = set()
    checked = Counter()
    decoder = {"decoding_plugin": "pylibjpeg"} if sop_class == CTImageStorage else {}
    for (source_path, _), output_path in zip(inputs, outputs):
        before, after = dcmread(source_path), dcmread(output_path)
        checked.update(_check_headers(before, after, profile, identifiers, forward, reverse))
        assert before.PixelData == after.PixelData
        av, bv = pixel_array(before, **decoder), pixel_array(after, **decoder)
        assert av.shape == (before.Rows, before.Columns)
        assert av.dtype == bv.dtype
        np.testing.assert_array_equal(av, bv)
        av, bv = apply_modality_lut(av, before), apply_modality_lut(bv, after)
        np.testing.assert_array_equal(av, bv)
        np.testing.assert_array_equal(apply_voi_lut(av, before), apply_voi_lut(bv, after))

        output_series[after.SeriesInstanceUID] += 1
        output_studies.add(after.StudyInstanceUID)
        output_sops.add(after.SOPInstanceUID)
        output_frames.add(after.FrameOfReferenceUID)
        output_uids.update(
            value for element in after.iterall() if element.VR == "UI"
            for value in _values(element)
        )
        output_references.extend(
            value for element in after.iterall() if element.keyword == "ReferencedSOPInstanceUID"
            for value in _values(element)
        )

    assert len(output_studies) == len(output_frames) == 1
    assert len(output_sops) == count and sorted(output_series.values()) == series_sizes
    if profile == "enhanced":
        assert not output_uids.intersection(forward), "an original mapped UID survived in the header"
    assert checked["IrradiationEventUID"] == event_occurrences
    assert sum(reference in output_sops for reference in output_references) == internal_references
    if sop_class == MRImageStorage:
        a, b = read_image(str(source)), read_image(str(output))
        assert a.shape == b.shape == (94, 512, 512)
        assert a.spacing == b.spacing == (2.5, 0.5, 0.5)
        assert a.origin == b.origin
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    assert _manifest(fixture) == source_manifest


def test_real_ct_media_root_rejects_dicomdir(tmp_path):
    source = Path(DATA_ROOT) / "2_skull_ct"
    before = _manifest(source)
    index = dcmread(source / "DICOMDIR")
    references = [item.ReferencedFileID for item in index.DirectoryRecordSequence
                  if "ReferencedFileID" in item]
    assert len(references) == 304
    assert sum(source.joinpath(*reference).is_file() for reference in references) == 303

    output = tmp_path / "export"
    stats = anonymize_dicom_dir(str(source), str(output), profile="enhanced")
    assert stats["file_count"] == stats["error_count"] == 1
    assert stats["success_count"] == 0
    [(path, message)] = stats["errors"]
    assert Path(path) == source / "DICOMDIR"
    assert "DICOMDIR" in message
    assert not list(output.iterdir())
    assert _manifest(source) == before
