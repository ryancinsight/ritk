"""Compare a local AsterSlice export of the downloaded TCIA MRI-DIR MR series.

Pass the Basic and Strict export folders produced from the 94-instance fixture.
The optional browser check uses a packaged workstation and a loopback driver.
"""
import argparse
from pathlib import Path
import shutil
import tempfile



SOURCE = Path(__file__).resolve().parents[1] / "test_data/2_head_mri_t2/DICOM"
ENTITY_UIDS = ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID")
TEMPORAL = ("StudyDate", "StudyTime", "SeriesDate", "SeriesTime",
            "AcquisitionDate", "AcquisitionTime", "ContentDate", "ContentTime")
DESCRIPTIONS = ("StudyDescription", "SeriesDescription")


def ordered(folder):
    # Directory export serializes in source-path order and assigns neutral names.
    files = sorted(folder.glob("*.dcm"))
    if len(files) != 94:
        raise ValueError("expected the complete 94-instance MRI-DIR series")
    return files


def verify_copy(source, folder, strict):
    from pydicom import dcmread

    files = ordered(folder)
    if [p.name for p in files] != [f"{n:06}.dcm" for n in range(1, 95)]:
        raise ValueError("numbered export filenames differ")
    # A referenced instance must get the same UID as its target wherever it appears.
    mappings = {}
    instances = set()

    def check_uid(old, new):
        assert new and new != old
        assert mappings.setdefault(old, new) == new

    for original, copy in zip(source, files, strict=True):
        a, b = dcmread(original), dcmread(copy)
        assert b.PatientName == "ANONYMOUS" and b.PatientID == "ANON001"
        assert b.PatientID != a.PatientID and b.PatientName != a.PatientName
        instances.add(a.SOPInstanceUID)
        for key in ENTITY_UIDS:
            if key not in a:
                assert key not in b
                continue
            check_uid(a.get(key), b.get(key))
        for sequence in ("ReferencedImageSequence", "ReferencedPerformedProcedureStepSequence"):
            assert len(a[sequence].value) > 0
            for old, new in zip(a[sequence].value, b[sequence].value, strict=True):
                assert new.ReferencedSOPClassUID == old.ReferencedSOPClassUID
                check_uid(old.ReferencedSOPInstanceUID, new.ReferencedSOPInstanceUID)
        assert b.file_meta.MediaStorageSOPInstanceUID == b.SOPInstanceUID
        assert a.file_meta.TransferSyntaxUID == b.file_meta.TransferSyntaxUID
        assert a.SOPClassUID == b.SOPClassUID == b.file_meta.MediaStorageSOPClassUID
        for tag in TEMPORAL + DESCRIPTIONS:
            if strict:
                assert not b.get(tag)
            else:
                assert a.get(tag) == b.get(tag)
        if strict:
            assert all(not elem.tag.is_private for elem in b.iterall())
        assert a.ImagePositionPatient == b.ImagePositionPatient
        assert a.ImageOrientationPatient == b.ImageOrientationPatient
        assert a.PixelSpacing == b.PixelSpacing
        assert (a.Rows, a.Columns) == (512, 512) == (b.Rows, b.Columns)
        assert a.PixelData == b.PixelData
        assert (a.pixel_array == b.pixel_array).all()
    assert len(mappings) == len(set(mappings.values()))
    assert len(instances) == len(files)


def verify_display(bundle, driver, groups, screenshot):
    from browser_protocol import StaticServer, WebDriverClient

    client = WebDriverClient(driver, 60)
    client.create_session("chrome", 1000, headless=True)
    try:
        client.set_window_rect(1440, 900)
        client.set_timeouts(30000)
        with tempfile.TemporaryDirectory(dir=bundle) as folder, StaticServer(bundle) as origin:
            probe = Path(folder) / "workstation_pixels.js"
            shutil.copyfile(Path(__file__).with_name(probe.name), probe)
            url = origin + probe.relative_to(bundle).as_posix()
            reference = None
            for name, files in groups:
                client.navigate(origin)
                wait(client, "!!window.asterSlice && !window.asterSlice.state().loaded")
                client.send_file_paths(client.find("#file-input"), files)
                wait(client, "window.asterSlice.state().counts.join(',') === '94,512,512' && window.asterSlice.state().indices[0] === 47")
                planes = client.execute_async("""
                    const [url,done] = arguments;
                    requestAnimationFrame(() => requestAnimationFrame(() => import(url)
                        .then(m => m.displayedPlanes()).then(done, e => done({error:String(e)}))));
                """, [url])
                assert len(planes["planes"]) == 3
                if reference is None:
                    reference = planes
                else:
                    assert planes == reference, f"{name} changed the displayed MRI planes"
                if name == "Strict" and screenshot:
                    screenshot.write_bytes(client.screenshot())
    finally:
        client.close()


def wait(client, condition):
    assert client.execute_async("""
        const done=arguments[arguments.length-1], end=performance.now()+25000;
        function poll() { if (""" + condition + """) done(true);
        else if(performance.now()>end) done(false); else setTimeout(poll,50); }
        poll();
    """), "workstation did not reach the expected study state"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("basic", type=Path)
    ap.add_argument("strict", type=Path)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--driver-url")
    ap.add_argument("--screenshot", type=Path, help="write the reopened Strict phantom screen")
    args = ap.parse_args()
    if bool(args.bundle) != bool(args.driver_url) or (args.screenshot and not args.bundle):
        ap.error("the browser comparison needs both --bundle and --driver-url")
    source = ordered(SOURCE)
    if not __debug__:
        ap.error("verification requires Python assertions to be enabled")
    for folder, strict in [(args.basic, False), (args.strict, True)]:
        verify_copy(source, folder, strict)
    if args.bundle:
        groups = [("Source", source), ("Basic", ordered(args.basic.resolve())),
                  ("Strict", ordered(args.strict.resolve()))]
        verify_display(args.bundle.resolve(), args.driver_url, groups, args.screenshot)
    print("Verified both 94-instance exports.")


if __name__ == "__main__":
    main()
