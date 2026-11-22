"""Exercise the packaged AsterSlice entry points against the native workspace geometry."""
from __future__ import annotations

import argparse
from pathlib import Path

from browser_protocol import BrowserRuntimeError, StaticServer, WebDriverClient


def verify(client: WebDriverClient, origin: str, study: Path) -> None:
    def require(expression: str, description: str) -> None:
        if client.execute("return " + expression) is not True:
            raise BrowserRuntimeError(description)

    def wait(expression: str) -> None:
        client.set_timeouts(30_000)
        passed = client.execute_async(
            "const done = arguments[arguments.length - 1];"
            "const end = performance.now() + 25000;"
            "function check() { if (" + expression + ") done(true);"
            "else if (performance.now() >= end) done(false);"
            "else window.setTimeout(check, 50); } check();"
        )
        if not passed:
            raise BrowserRuntimeError(f"workspace state did not settle: {expression}")

    def click(selector: str) -> None:
        client.click(client.find(selector))

    client.set_window_rect(1440, 900)
    client.navigate(origin)
    require('document.title === "AsterSlice"', "directory index opens the wrong application")
    wait("window.metisGallery?.sample().mounted")
    require('document.querySelector("#series-bar").offsetWidth === 276', "native series width")
    require('document.querySelector(".menubar").offsetHeight === 30', "native menu height")
    require('document.querySelector(".toolbar").offsetHeight === 48', "native toolbar height")
    require('document.querySelector("footer").offsetHeight === 26', "native status bar height")
    require("document.documentElement.scrollHeight === innerHeight", "desktop page overflows")
    client.send_file_paths(client.find("#file-input"), sorted(study.resolve().glob("*.dcm")))
    wait("document.querySelector('#position-axial').textContent === '48 / 94'")
    require("document.querySelector('#series-size').textContent === '94 images'", "MRI series count")
    require("""document.querySelector('#series-thumb').getContext('2d')
        .getImageData(0,0,64,64).data.some((value,i) => i%4 !== 3 && value > 0)""",
            "loaded series thumbnail is blank")
    click("#series-toggle")
    require("document.querySelector('#series-bar').hidden", "series bar did not close")
    click("#series-toggle")
    click("#focus-axial")
    require("document.querySelector('#focus-coronal').getClientRects().length === 0", "maximize pane")
    click("#focus-axial")
    require("document.querySelector('#focus-coronal').getClientRects().length === 1", "restore panes")
    click('#quick-tools [data-tool="Length"]')
    wait("document.querySelector('#active-tool').textContent === 'Active tool: Length'")
    click("#tools-menu")
    click('#tool-buttons [data-tool="Angle"]')
    wait('document.querySelector(\'#quick-tools [data-tool="Angle"]\').ariaPressed === "true"')
    client.set_window_rect(500, 820)
    require("document.documentElement.scrollWidth === innerWidth", "narrow page overflows")
    require("document.querySelector('#series-bar').hidden", "narrow sidebar obscures the image")
    client.navigate(origin + "gallery.html?projection=mip")
    wait("window.metisGallery?.sample().projection_mode === 'mip'")
    require("!document.querySelector('#projection-view').hidden", "projection deep link")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("study", type=Path)
    parser.add_argument("--driver-url", required=True)
    args = parser.parse_args()
    client = WebDriverClient(args.driver_url, 30)
    client.create_session("chrome", 1000, headless=True)
    try:
        with StaticServer(args.bundle.resolve()) as origin:
            verify(client, origin, args.study)
    finally:
        client.close()
