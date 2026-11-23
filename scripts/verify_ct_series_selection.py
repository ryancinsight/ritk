"""Check the initially displayed acquisition on three local copies of the five-series CT."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import tempfile

from browser_protocol import StaticServer, WebDriverClient
from pydicom import dcmread
from pydicom.errors import InvalidDicomError

from verify_anonymized_mri import wait
from verify_study_folder import choose


def verify_inventory(folder):
    counts = Counter()
    for path in sorted(folder.iterdir()):
        if path.is_file():
            try:
                ds = dcmread(path, stop_before_pixels=True)
            except InvalidDicomError:
                continue
            if ds.get("Modality") == "CT":
                counts[int(ds.SeriesNumber)] += 1
    if counts != {1: 2, 2: 34, 3: 67, 601: 111, 602: 89}:
        raise ValueError("expected the complete five-series CT")


def verify(bundle, driver, folders):
    client = WebDriverClient(driver, 90)
    client.create_session("chrome", 1000, headless=True)
    try:
        client.set_window_rect(1440, 900)
        with tempfile.TemporaryDirectory(dir=bundle) as scratch, StaticServer(bundle) as origin:
            probe = Path(scratch) / "workstation_pixels.js"
            shutil.copyfile(Path(__file__).with_name(probe.name), probe)
            url = origin + probe.relative_to(bundle).as_posix()
            reference = None
            for label, folder in zip(("Source", "Basic", "Strict"), folders, strict=True):
                verify_inventory(folder)
                client.navigate(origin)
                wait(client, "!!window.asterSlice && !window.asterSlice.state().loaded")
                choose(client, folder)
                wait(client, "window.asterSlice.state().loaded && window.asterSlice.state().series === 5")
                shape = client.execute("return window.asterSlice.state().counts")
                if shape != [34, 512, 512]:
                    raise ValueError(f"{label} opened {shape}, expected the 34-image acquisition")
                regions = client.execute_async("""
                    const [url, done] = arguments;
                    requestAnimationFrame(() => requestAnimationFrame(() => import(url)
                        .then(m => m.displayedPlanes()).then(done, e => done({error: String(e)}))));
                """, [url])
                if len(regions["planes"]) != 3 or reference is not None and regions != reference:
                    raise ValueError(f"{label} displayed different image regions")
                reference = regions
                print(f"{label}: 5 series, 34-image acquisition, three displayed planes verified")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("basic", type=Path)
    parser.add_argument("strict", type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--driver-url", required=True)
    args = parser.parse_args()
    verify(args.bundle.resolve(), args.driver_url, (args.source, args.basic, args.strict))
