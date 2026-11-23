"""Exercise the packaged AsterSlice entry points against the native workspace geometry."""
from __future__ import annotations

import argparse
import json
import tempfile
from urllib.parse import urlencode
from browser_local_box_provenance import copy_bundle
from browser_navigation import verify_navigation
from pathlib import Path

from browser_protocol import BrowserRuntimeError, StaticServer, WebDriverClient


def verify(client: WebDriverClient, origin: str, study: Path, *, comparison: Path | None = None,
           query: str = "", output: Path | None = None, embed: str | None = None) -> None:
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

    def click_at(x: int, y: int) -> None:
        client.perform_actions([{"type": "pointer", "id": "mouse", "parameters": {"pointerType": "mouse"},
            "actions": [{"type": "pointerMove", "x": x, "y": y, "origin": "viewport"},
                        {"type": "pointerDown", "button": 0}, {"type": "pointerUp", "button": 0}]}])

    client.set_window_rect(1440, 900)
    client.navigate(origin + "?" + query)
    require('document.title === "AsterSlice"', "directory index opens the wrong application")
    require('document.querySelectorAll("canvas").length === 1 && !!document.querySelector("#workstation")',
            "expected the native workstation framebuffer at the application root")
    wait("window.asterSlice?.state().width === innerWidth")
    state = "window.asterSlice.state()"
    require(state + ".panes[0][0] === 276", "native study rail width")
    require(state + ".panes[0][1] === 78", "native menu and toolbar height")
    files = sorted(study.resolve().glob("*.dcm"))
    if not files:
        raise BrowserRuntimeError("study contains no DICOM instances")
    client.send_file_paths(client.find("#file-input"), files)
    wait(state + ".loaded")
    require(state + f".counts[0] === {len(files)}", "decoded axial slice count")
    require(state + f".indices[0] === {len(files) // 2}", "native initial slice")
    if output:
        output.mkdir(exist_ok=True, parents=True)
        (output / "workstation.png").write_bytes(client.screenshot())
    # Collect the displayed samples away from overlay text and center crosshairs.
    samples = client.execute_async("""
        const done = arguments[arguments.length-1], c = document.querySelector('#workstation');
        c.toBlob(async blob => {
          const image = await createImageBitmap(blob);
          const capture = document.createElement('canvas'); capture.width=c.width; capture.height=c.height;
          const ctx=capture.getContext('2d');ctx.drawImage(image,0,0);
          const state=window.asterSlice.state();
          done(state.images.map(([left,top,width,height],axis) => {
            const [x,y,w,h]=state.panes[axis];
            const [sw,sh]=[[512,512],[512,94],[512,94]][axis], pixels=[];
            for(const fx of [.32,.38,.62,.69]) for(const fy of [.31,.36,.61,.68]){
              const px=Math.floor(left+width*fx),py=Math.floor(top+height*fy);
              if(px<x || px>=x+w || py<y || py>=y+h) continue;
              pixels.push([Math.floor((px+.5-left)/(width/sw)), Math.floor((py+.5-top)/(height/sh)),
                Array.from(ctx.getImageData(px,py,1,1).data)]);
            }
            return pixels;
          }));
          image.close();
        }, "image/png");
    """)
    if len(samples) != 3 or any(not plane for plane in samples):
        raise BrowserRuntimeError("missing visible orthogonal image samples")
    x, y, w, h = client.execute("return " + state + ".panes[0]")
    cw, ch = client.execute("return [innerWidth,innerHeight]")
    center = (x + w // 2 - cw // 2, y + h // 2 - ch // 2)
    canvas = client.find("#workstation")
    home, end = chr(0xe011), chr(0xe010)
    verify_navigation(client, canvas, state, wait)
    initial_slice = client.execute("return " + state + ".indices[0]")
    client.wheel(canvas, center, (0, 120))
    wait(f"Math.abs({state}.indices[0] - {initial_slice}) === 1")
    click_at(555, 53)
    wait(state + ".tool === 'Length'")
    client.pointer_drag(canvas, (center[0]-20, center[1]), (center[0]+20, center[1]))
    wait(state + ".annotations === 1 && " + state + ".measurement > 0")
    click_at(345, 53)
    before = client.execute("return " + state + ".window")
    client.pointer_drag(canvas, center, (center[0] + 18, center[1] + 20))
    if client.execute("return " + state + ".window") == before:
        raise BrowserRuntimeError("window and level did not change")
    click_at(480, 53)
    client.pointer_drag(canvas, center, (center[0], center[1] + 20))
    wait(state + ".zoom !== 1")
    client.send_keys(canvas, "r")
    wait("!" + state + ".orientationIdentity")
    client.send_keys(canvas, " ")
    wait(state + ".cine")
    client.execute("window.cineSlice = " + state + ".indices[0]")
    wait(state + ".indices[0] !== window.cineSlice")
    client.send_keys(canvas, " ")
    wait("!" + state + ".cine")
    client.send_keys(canvas, home)
    wait(state + ".indices[0] === 0")
    client.send_keys(canvas, end)
    wait(state + f".indices[0] === {len(files) - 1}")
    if comparison:
        members = sorted(comparison.resolve().glob("*.dcm"))
        if len(members) < 2:
            raise BrowserRuntimeError("two CT instances required")
        client.send_file_paths(client.find("#file-input"), [*files, *members[:2]])
        wait(state + ".series === 2")
        client.send_keys(canvas, chr(0xe034))  # F4 opens the native series dialog.
        # Select both real series in the native series dialog.
        client.send_keys(canvas, " " + chr(0xe015) + " " + chr(0xe007))
        wait(state + ".panels.length === 2")
        require(state + ".panels.map(p=>p[1]).sort((a,b)=>a-b).join(',') === '2,94'",
                "independent CT and MR volumes")
        client.send_keys(canvas, chr(0xe004))  # Tab advances the active comparison panel.
        wait(state + ".panel === 1")
        # The picker assigns both catalog entries in order. At the matching
        # endpoint the series and its current image must both stay selected.
        for panel, endpoint in [(1, end), (0, home)]:
            baseline = client.execute("return " + state + ".panels")
            client.send_keys(canvas, endpoint)
            observed = client.execute("return " + state + ".panels")
            if observed != baseline:
                raise BrowserRuntimeError(f"series endpoint moved an image in panel {panel}: {baseline} -> {observed}")
            client.send_keys(canvas, chr(0xe004))
            wait(state + f".panel === {1-panel}")
        # Switch the second panel to the first series, then restore the last.
        first, last = client.execute("return " + state + ".panels")
        client.send_keys(canvas, home)
        wait(f"{state}.panels[1][1] === {first[1]}")
        client.send_keys(canvas, end)
        wait(f"{state}.panels[1][1] === {last[1]}")
        if client.execute("return " + state + ".panels[0]") != first:
            raise BrowserRuntimeError("changing series in panel two moved the first panel")
        initial = client.execute("return " + state + ".panels")
        index = next(i for i, (_, count) in enumerate(initial) if count == len(files))
        px, py, pw, ph = client.execute("return " + state + f".panes[{index}]")
        client.wheel(canvas, (px + pw//2 - cw//2, py + ph//2 - ch//2), (0, 120))
        wait(f"Math.abs({state}.panels[{index}][0] - {initial[index][0]}) === 1")
        require(f"{state}.panels[{1-index}][0] === {initial[1-index][0]}", "wheel changed the other series")
        if output:
            (output / "comparison.png").write_bytes(client.screenshot())
    client.set_window_rect(500, 820)
    wait(state + ".width === innerWidth && " + state + ".panes[0][0] === 0")
    require("document.documentElement.scrollWidth === innerWidth", "narrow page overflows")
    client.navigate(origin + "gallery.html?projection=mip")
    wait("window.asterSlice?.state().projection === true")
    if embed is None:
        return
    # Independently render the original planes via the published embedding API.
    client.navigate(embed)
    wait("document.getElementById('ready')?.textContent === 'Ready.'")
    client.send_file_paths(client.find("#file-input"), files)
    wait("document.querySelectorAll('canvas[data-ritk-load-state=ready]').length === 3")
    oracle = Path(__file__).resolve().parents[1] / "docs/manual/images/dicom-metis-real-browser-mri-edge.json"
    views = json.loads(oracle.read_text())["views"]
    for axis, name in enumerate(("axial", "coronal", "sagittal")):
        result = client.execute_async("""
            const [name, samples, done] = arguments;
            const c=document.getElementById('ritk-snap-'+name),ctx=c.getContext('2d');
            const image=ctx.getImageData(0,0,c.width,c.height).data;
            const equal=samples.every(([x,y,rgba]) => rgba.every((v,i)=>v===image[(y*c.width+x)*4+i]));
            crypto.subtle.digest('SHA-256',image).then(bytes=>
              done([equal, Array.from(new Uint8Array(bytes),v=>v.toString(16).padStart(2,'0')).join('')]));
        """, [name, samples[axis]])
        if result != [True, views[name]["rgba_sha256"]]:
            raise BrowserRuntimeError(f"{name} screen samples or reference plane differ: {result}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("study", type=Path)
    parser.add_argument("--driver-url", required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--engine", choices=("chromium","firefox","webkit"), default="chromium")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--renderer", default="raster")
    parser.add_argument("--projection", default="")
    parser.add_argument("--layout", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    client = WebDriverClient(args.driver_url, 60)
    client.create_session({"chromium":"chrome", "firefox":"firefox", "webkit":"safari"}[args.engine],
                          args.scale * 1000, headless=args.engine != "webkit")
    query = urlencode({k:v for k,v in {"renderer":args.renderer, "projection":args.projection,
                                         "layout":args.layout}.items() if v})
    try:
        with tempfile.TemporaryDirectory(dir=bundle) as temporary:
            fixture = Path(temporary) / "embedded"
            copy_bundle(bundle, fixture)
            with StaticServer(bundle) as origin:
                for cycle in range(args.cycles):
                    verify(client, origin, args.study, comparison=args.comparison, query=query,
                           output=args.output, embed=origin + fixture.relative_to(bundle).as_posix() + "/embed.html")
    finally:
        client.close()
