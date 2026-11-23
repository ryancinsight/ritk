"""Compare the three complete Safari image regions against Chromium at the same viewport."""
import argparse
import json
from pathlib import Path

from browser_protocol import WebDriverClient


def wait(client, condition):
    settled = client.execute_async(
        "const done=arguments[0], start=performance.now(); function poll(){"
        "if (" + condition + ") done(true);"
        "else if(performance.now()-start>20000) done(false); else requestAnimationFrame(poll); } poll();"
    )
    if not settled:
        raise RuntimeError("Reference workstation did not reach " + condition)


def read_display(client, origin, files, width, height):
    client.navigate(origin + "index.html")
    actual_width, actual_height = client.execute("return [innerWidth,innerHeight]")
    outer_width, outer_height = client.execute("return [outerWidth,outerHeight]")
    client.set_window_rect(outer_width + width - actual_width, outer_height + height - actual_height)
    wait(client, f"window.asterSlice?.state().width === {width} && window.asterSlice.state().height === {height}")
    x, y, w, h = client.execute("return window.asterSlice.state().panes[0]")
    client.perform_actions([{"type": "pointer", "id": "mouse", "parameters": {"pointerType": "mouse"},
        "actions": [{"type": "pointerMove", "origin": "viewport", "x": x + w//2, "y": y + h//2},
                    {"type": "pointerDown", "button": 0}, {"type": "pointerUp", "button": 0}]}])
    client.send_file_paths(client.find("#file-input"), files)
    wait(client, "window.asterSlice.state().counts.join(',') === '94,512,512' && "
         "window.asterSlice.state().indices.join(',') === '47,256,256'")
    result = client.execute_async(
        "const done=arguments[0]; import('./workstation_pixels.js').then(m=>m.displayedPlanes())"
        ".then(done,e=>done({error:String(e)}));"
    )
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def compare(attachments, study, output, origin, driver):
    manifest = json.loads((attachments / "manifest.json").read_text())
    filename, = [item["exportedFileName"] for test in manifest for item in test["attachments"]
                 if item["suggestedHumanReadableName"].split(".")[0] == "Displayed MRI planes"]
    safari = json.loads((attachments / filename).read_text())
    files = sorted(study.resolve().glob("*.dcm"))
    if len(files) != 94 or sum(path.stat().st_size for path in files) != 49_807_236:
        raise ValueError("The display reference requires all 94 public MRI instances")
    client = WebDriverClient(driver, 30)
    client.create_session("chrome", 1000, headless=True)
    try:
        chromium = read_display(client, origin, files, safari["width"], safari["height"])
    finally:
        client.close()
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps({"safari": safari, "chromium": chromium}, indent=2) + "\n")
    if len(safari["planes"]) != 3 or safari != chromium:
        raise ValueError(f"Displayed MRI plane geometry or RGBA bytes differ: {output.read_text()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attachments", type=Path)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--origin", default="http://127.0.0.1:17823/")
    parser.add_argument("--driver", default="http://127.0.0.1:9515")
    args = parser.parse_args()
    compare(args.attachments, args.study, args.output, args.origin, args.driver)
