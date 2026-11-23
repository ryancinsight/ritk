"""Exercise the Open Study toolbar with a local directory and verify the decoded catalog."""
import argparse
from pathlib import Path

from browser_protocol import StaticServer, WebDriverClient
from verify_anonymized_mri import wait


def verify(url, folder, series, slices, driver):
    client = WebDriverClient(driver, 90)
    client.create_session("chrome", 1000, headless=True)
    try:
        client.set_window_rect(1440, 900)
        client.navigate(url)
        wait(client, "!!window.asterSlice")
        # Record which native chooser the real toolbar activates. WebDriver supplies
        # the chosen directory so the OS dialog does not need manual interaction.
        client.execute("""
            document.addEventListener('click', e => {
                if (e.target.type !== 'file') return;
                window.openedPicker = e.target.id;
                e.preventDefault();
            }, true);
        """)
        width, height = client.execute("return [innerWidth,innerHeight]")
        point = (62 - width // 2, 53 - height // 2)
        client.pointer_drag(client.find("#workstation"), point, point)
        picker = client.execute("return window.openedPicker")
        if not picker or not client.execute("return document.getElementById(arguments[0]).webkitdirectory", [picker]):
            raise AssertionError("Open Study requested individual files instead of a folder")
        client.send_file_paths(client.find("#" + picker), [folder.resolve(strict=True)])
        wait(client, "window.asterSlice.state().loaded || !document.querySelector('#notice').hidden")
        if not client.execute("return window.asterSlice.state().loaded"):
            raise AssertionError("the workstation rejected the folder")
        state = client.execute("return window.asterSlice.state()")
        if (state["series"], state["counts"]) != (series, [slices, 512, 512]):
            raise AssertionError("folder did not open the expected series and volume")
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
    args = parser.parse_args()
    if args.bundle:
        with StaticServer(args.bundle.resolve()) as origin:
            verify(origin, args.folder, args.series, args.slices, args.driver_url)
    else:
        verify(args.url, args.folder, args.series, args.slices, args.driver_url)
