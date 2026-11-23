"""Build the AsterSlice web app using the pinned Métis consumer validation."""
import argparse
import hashlib
import pathlib
import shutil
import sys


def package(metis: pathlib.Path, modules: pathlib.Path, output: pathlib.Path) -> None:
    web = pathlib.Path(__file__).resolve().parents[1] / "crates/ritk-snap/web"
    sys.path.insert(0, str(metis.resolve() / "scripts"))
    import browser as metis_browser

    # Métis checks the module header, byte limits and same-origin policy.
    metis_browser.OUTPUT = output.resolve()
    metis_browser.package_consumer(modules.resolve(), web / "workstation")
    # The mounted host creates a picture element even while its header is hidden.
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    for name in ("metis-mark.svg", "metis-mark.png"):
        shutil.copyfile(metis / "examples/browser/assets" / name, assets / name)
    # A cached entry script must not bind an older picker or WASM module to a new page.
    digest = hashlib.sha256()
    for asset in ("gallery.js", "gallery.css", "consumer/ritk_snap.js", "consumer/ritk_snap_bg.wasm"):
        with (output / asset).open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    version = digest.hexdigest()
    page = output / "gallery.html"
    html = page.read_text(encoding="utf-8")
    for asset in ("gallery.js", "gallery.css"):
        html = html.replace(f'"./{asset}"', f'"./{asset}?v={version}"')
    page.write_text(html, encoding="utf-8")
    shutil.copyfile(page, output / "index.html")
    # Also retire the previous app's asset when rebuilding an existing output.
    (output / "workspace.js").unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metis-root", required=True, type=pathlib.Path)
    parser.add_argument("modules", type=pathlib.Path, help="wasm-bindgen output directory")
    parser.add_argument("output", type=pathlib.Path, help="web application directory")
    args = parser.parse_args()
    package(args.metis_root, args.modules, args.output)
