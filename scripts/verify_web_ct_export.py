"""Exercise browser downloads and reopen both profiles of the retained five-series CT locally."""
import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from zipfile import ZipFile

from browser_protocol import WebDriverClient
from pydicom import dcmread
from pydicom.pixels import pixel_array
from numpy import array_equal

from verify_anonymized_mri import TEMPORAL, DESCRIPTIONS, wait
from verify_study_folder import choose


def click(client, x, y):
    width, height = client.execute("return [innerWidth,innerHeight]")
    point = (int(x - width // 2), int(y - height // 2))
    client.pointer_drag(client.find("#workstation"), point, point)
    client.release_actions()


def inventory(source, folder, profile):
    mapping, targets, references = {}, set(), []
    counts = Counter()
    pixels = 0
    for number, path in enumerate(sorted((source / "DICOM").iterdir()), 1):
        a = dcmread(path)
        b = dcmread(folder / f"{number:06}.dcm")
        assert b.PatientName == "ANONYMOUS" and b.PatientID == "ANON001"
        assert b.PatientName != a.PatientName and b.PatientID != a.PatientID
        assert b.preamble == bytes(128)
        assert b.file_meta.MediaStorageSOPInstanceUID == b.SOPInstanceUID
        assert a.file_meta.TransferSyntaxUID == b.file_meta.TransferSyntaxUID
        for key in ("SOPClassUID", "SeriesNumber", "ImagePositionPatient", "ImageOrientationPatient",
                    "PixelSpacing", "Rows", "Columns", "RescaleSlope", "RescaleIntercept"):
            assert a.get(key) == b.get(key)
        for key in TEMPORAL + DESCRIPTIONS:
            assert (not b.get(key)) if profile == "Strict" else a.get(key) == b.get(key)
        old_refs, new_refs = ([e.value for e in ds.iterall() if e.keyword == "ReferencedSOPInstanceUID"]
                              for ds in (a, b))
        pairs = list(zip(old_refs, new_refs, strict=True))
        references.extend(new_refs)
        for key in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
                    "FrameOfReferenceUID", "IrradiationEventUID"):
            if key in a:
                pairs.append((a.get(key), b.get(key)))
        for old, new in pairs:
            assert new.is_valid and old != new and mapping.setdefault(old, new) == new
        targets.add(b.SOPInstanceUID)
        counts[int(b.SeriesNumber)] += 1
        assert a.PixelData == b.PixelData
        original, copy = (pixel_array(ds, decoding_plugin="pylibjpeg") for ds in (a, b))
        assert original.dtype == copy.dtype and array_equal(original, copy)
        pixels += original.size
    assert counts == {1: 2, 2: 34, 3: 67, 601: 111, 602: 89}
    assert len(targets) == 303 and sum(ref in targets for ref in references) == 101
    assert len(mapping) == len(set(mapping.values())) and pixels == 79_518_234
    return set(mapping.values())


def displayed(client):
    source = Path(__file__).with_name("workstation_pixels.js").read_text().replace("export ", "")
    return client.execute_async(source + """
        const done = arguments[arguments.length - 1];
        requestAnimationFrame(() => requestAnimationFrame(() => displayedPlanes().then(done)));
    """)


def verify(url, source, driver):
    client = WebDriverClient(driver, 120)
    client.create_session("chrome", 1000, headless=True)
    try:
        client.set_window_rect(1440, 900)
        previous = set()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            client._request("POST", client._session_path("goog/cdp/execute"), {
                "cmd": "Browser.setDownloadBehavior", "params": {"behavior": "allow", "downloadPath": str(root)}})
            for row, profile in enumerate(("Basic", "Strict")):
                client.navigate(url)
                wait(client, "!!window.asterSlice")
                choose(client, source)
                wait(client, "window.asterSlice.state().series === 5")
                assert client.execute("return asterSlice.state().counts") == [34, 512, 512]
                original = displayed(client)
                click(client, 300, 15)
                click(client, 320, 44 + 28 * row)
                archive = root / f"AsterSlice-{profile}.zip"
                deadline = monotonic() + 15
                while not archive.exists() and monotonic() < deadline:
                    sleep(.1)
                assert archive.exists(), f"{profile} menu did not produce a ZIP download"
                copy = root / profile
                with ZipFile(archive) as z:
                    assert z.namelist() == [f"{n:06}.dcm" for n in range(1,304)]
                    z.extractall(copy)
                assert client.execute("return asterSlice.state().exportAvailable")
                identifiers = inventory(source, copy, profile)
                assert identifiers.isdisjoint(previous)
                previous = identifiers
                click(client, 300, 15)
                click(client, 320, 156)
                wait(client, "!asterSlice.state().exportAvailable && asterSlice.state().series === 5")
                assert displayed(client) == original
                client.navigate(url)
                wait(client, "!!window.asterSlice")
                choose(client, copy)
                wait(client, "asterSlice.state().series === 5")
                assert displayed(client) == original
                print(f"{profile}: verified download, 303 instances, 101 references, 79,518,234 pixels and both reopen paths", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8108/")
    parser.add_argument("--driver-url", required=True)
    args = parser.parse_args()
    verify(args.url, args.source, args.driver_url)
