"""Exercise the Open Study toolbar with a local directory and verify the decoded catalog."""
import argparse
from pathlib import Path

from browser_protocol import StaticServer, WebDriverClient
from verify_anonymized_mri import wait


def choose(client, path, *, individual=False):
    # Record which native chooser the real toolbar activates. WebDriver supplies
    # the chosen directory so the OS dialog does not need manual interaction.
    client.execute("""
        window.openedPicker = undefined;
        window.pickerProbe = new AbortController();
        document.addEventListener('click', e => {
            if (e.target.type !== 'file') return;
            window.openedPicker = e.target.id;
            e.preventDefault();
        }, {capture: true, signal: window.pickerProbe.signal});
    """)
    width, height = client.execute("return [innerWidth,innerHeight]")
    point = (62 - width // 2, 53 - height // 2)
    if individual:
        client.perform_actions([{"type": "key", "id": "picker-shift",
                                 "actions": [{"type": "keyDown", "value": chr(0xe008)}]}])
    try:
        client.pointer_drag(client.find("#workstation"), point, point)
    finally:
        client.release_actions()
    picker = client.execute("window.pickerProbe.abort(); return window.openedPicker ?? null")
    directory = client.execute("return document.getElementById(arguments[0])?.webkitdirectory", [picker])
    if not picker or directory != (not individual):
        raise AssertionError("Open Study requested the wrong kind of file selection")
    client.send_file_paths(client.find("#" + picker), [path.resolve(strict=True)])


def verify(url, folder, series, slices, driver, individual_file=None):
    client = WebDriverClient(driver, 90)
    client.create_session("chrome", 1000, headless=True)
    try:
        client.set_window_rect(1440, 900)
        client.navigate(url)
        wait(client, "!!window.asterSlice")
        choose(client, folder)
        wait(client, "window.asterSlice.state().loaded || !document.querySelector('#notice').hidden")
        if not client.execute("return window.asterSlice.state().loaded"):
            raise AssertionError("the workstation rejected the folder")
        state = client.execute("return window.asterSlice.state()")
        if (state["series"], state["counts"]) != (series, [slices, 512, 512]):
            raise AssertionError("folder did not open the expected series and volume")
        if individual_file is not None:
            choose(client, individual_file, individual=True)
            wait(client, "window.asterSlice.state().counts[0] === 1")
        versions = client.execute("""
            const urls = performance.getEntriesByType("resource").map(entry => new URL(entry.name));
            return ["gallery.css", "gallery.js", "ritk_snap.js", "ritk_snap_bg.wasm"].map(name =>
                urls.find(url => url.pathname.endsWith("/" + name))?.searchParams.get("v"));
        """)
        if not all(versions) or len(set(versions)) != 1:
            raise AssertionError("the workstation fetched assets from different or unversioned builds")
        print(f"Open Study folder verified: {series} series, {slices} initial slices")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--series", required=True, type=int)
    parser.add_argument("--slices", required=True, type=int)
    parser.add_argument("--url", default="http://127.0.0.1:8108/")
    parser.add_argument("--driver-url", default="http://127.0.0.1:9515")
    parser.add_argument("--bundle", type=Path, help="serve a packaged viewer locally during this check")
    parser.add_argument("--individual-file", type=Path, help="also Shift-click to open this single-slice instance")
    args = parser.parse_args()
    if args.bundle:
        with StaticServer(args.bundle.resolve()) as origin:
            verify(origin, args.folder, args.series, args.slices, args.driver_url, args.individual_file)
    else:
        verify(args.url, args.folder, args.series, args.slices, args.driver_url, args.individual_file)
