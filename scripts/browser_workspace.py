"""Exercise the packaged AsterSlice entry points against the native workspace geometry."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from urllib.parse import urlencode, urljoin
from browser_local_box_provenance import copy_bundle
from pathlib import Path
from workstation_admission import invalid_studies

from browser_protocol import BrowserRuntimeError, StaticServer, WebDriverClient


def verify(client: WebDriverClient, origin: str, study: Path, *, comparison: Path | None = None,
           query: str = "", output: Path | None = None, embed: str, cycles: int = 1) -> None:
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
    if len(files) != 94:
        raise BrowserRuntimeError(f"expected all 94 public MRI instances, got {len(files)}")
    expected_bytes = sum(file.stat().st_size for file in files)
    module = urljoin(embed, "workstation_pixels.js")

    def probe(method: str, *arguments):
        return client.execute_async("""
            const [url, method, args, done] = arguments;
            import(url).then(m => m[method](...args)).then(done, e => done({error:String(e)}));
        """, [module, method, list(arguments)])

    def capacity():
        return client.execute_async("""
            const done = arguments[arguments.length - 1];
            import('./consumer/ritk_snap.js').then(async m => done((await m.default()).memory.buffer.byteLength));
        """)

    client.execute("""
        window.studySelection = {generation:0};
        document.addEventListener('change', event => {
          if (!event.isTrusted || event.target.id !== 'file-input') return;
          const files = Array.from(event.target.files);
          window.studySelection = {generation:window.studySelection.generation+1,
            count:files.length, bytes:files.reduce((n,f)=>n+f.size,0)};
        }, true);
    """)
    x, y, w, h = client.execute("return " + state + ".panes[0]")
    cw, ch = client.execute("return [innerWidth,innerHeight]")
    center = (x + w // 2 - cw // 2, y + h // 2 - ch // 2)
    canvas = client.find("#workstation")
    history, observations = [], []
    for cycle in range(1, cycles + 1):
        initial_bytes = capacity()
        client.send_file_paths(client.find("#file-input"), files)
        wait(f"window.studySelection.generation === {cycle} && {state}.loaded && {state}.indices[0] === 47")
        selection = client.execute("return window.studySelection")
        if selection != {"generation": cycle, "count": 94, "bytes": expected_bytes}:
            raise BrowserRuntimeError(f"invalid trusted MRI selection: {selection}")
        require(state + ".counts.join(',') === '94,512,512'", "decoded MRI dimensions")
        sampled = probe("screenSamples")
        if not isinstance(sampled, list) or len(sampled) != 3 or any(not plane for plane in sampled):
            raise BrowserRuntimeError(f"missing visible orthogonal image samples: {sampled}")
        observations.append(sampled)
        loaded_bytes = capacity()
        if output and cycle == 1:
            output.mkdir(exist_ok=True, parents=True)
            (output / "workstation.png").write_bytes(client.screenshot())
        client.wheel(canvas, center, (0, 120))
        wait("Math.abs(" + state + ".indices[0] - 47) === 1")
        history.append({"selection":selection, "before":initial_bytes, "decoded":loaded_bytes,
                        "navigated":capacity(), "slice":client.execute("return "+state+".indices[0]")})
    if cycles >= 4:
        if output:
            (output / "consecutive-loads.json").write_text(json.dumps(history, indent=2)+"\n")
        if any(history[-1][key] != history[-2][key] for key in ("before","decoded","navigated")):
            raise BrowserRuntimeError(f"WebAssembly capacity changed after warmup: {history}")

    before_rejection = probe("screenSamples")
    if not isinstance(before_rejection, list) or len(before_rejection) != 3:
        raise BrowserRuntimeError(f"missing pre-rejection frame: {before_rejection}")
    with tempfile.TemporaryDirectory() as rejected:
        for name, invalid in invalid_studies(Path(rejected)):
            client.send_file_paths(client.find("#file-input"), invalid)
            wait(f"window.studySelection.count === {len(invalid)} && " +
                 "!!document.querySelector('#drop-zone[data-drop-state=rejected][data-byte-state=failed]')")
            if probe("screenSamples") != before_rejection:
                raise BrowserRuntimeError(f"{name} rejection changed a displayed plane")
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
    # Independently render the original planes via the published embedding API.
    client.navigate(embed)
    wait("document.getElementById('ready')?.textContent === 'Ready.'")
    client.send_file_paths(client.find("#file-input"), files)
    wait("document.querySelectorAll('canvas[data-ritk-load-state=ready]').length === 3")
    oracle = Path(__file__).resolve().parents[1] / "docs/manual/images/dicom-metis-real-browser-mri-edge.json"
    views = json.loads(oracle.read_text())["views"]
    for axis, name in enumerate(("axial", "coronal", "sagittal")):
        samples = [pixel for cycle in observations for pixel in cycle[axis]]
        result = probe("comparePlane", name, samples)
        if result != [True, views[name]["rgba_sha256"]]:
            raise BrowserRuntimeError(f"{name} screen samples or reference plane differ: {result}")

    cleanup = client.execute_async("""
        const done = arguments[arguments.length - 1];
        import('./consumer/ritk_snap.js').then(m => {
          const before = m.web_canvas_listener_count();
          m.stop_web_canvas();
          done([before, m.web_canvas_listener_count()]);
        });
    """)
    if not (cleanup[0] > 0 and cleanup[1] == 0):
        raise BrowserRuntimeError(f"embedded viewer listeners remained after stop: {cleanup}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("study", type=Path)
    parser.add_argument("--driver-url", required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--engine", choices=("chromium","firefox","webkit"), default="chromium")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--cycles", type=int, choices=range(1, 5), default=1)
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
            shutil.copy(Path(__file__).with_name("workstation_pixels.js"), fixture)
            with StaticServer(bundle) as origin:
                embedded_url = origin + fixture.relative_to(bundle).as_posix() + "/embed.html"
                for cycle in range(args.cycles):
                    verify(client, origin, args.study, comparison=args.comparison, query=query,
                           output=args.output, cycles=args.cycles if cycle == 0 else 1, embed=embedded_url)
    finally:
        client.close()
