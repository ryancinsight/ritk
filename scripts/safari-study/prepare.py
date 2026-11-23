"""Stage only the two published test studies and instrument a disposable packaged copy."""
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from browser_local_box_provenance import copy_bundle
from workstation_admission import invalid_studies

root = HERE.parents[1]
bundle = Path(sys.argv[1]).resolve(strict=True)
stage = Path(sys.argv[2])
stage.mkdir()
mr = sorted((root / "test_data/2_head_mri_t2/DICOM").glob("*.dcm"))
ct = sorted((root / "test_data/3_head_ct_mridir/DICOM").glob("*.dcm"))
assert len(mr) == 94 and len(ct) >= 2
for name, files in (("MRI", mr), ("Comparison", mr + ct[:2])):
    directory = stage / name
    directory.mkdir()
    for index, path in enumerate(files, 1):
        shutil.copyfile(path, directory / f"{index:08}.dcm")
list(invalid_studies(stage))
embedded = bundle / "embedded"
copy_bundle(bundle, embedded)
for page in (bundle / "index.html", bundle / "gallery.html", embedded / "embed.html"):
    content = page.read_text()
    page.write_text(content.replace("</body>", '<script type="module" src="./qualification.js"></script></body>'))
for directory in (bundle, embedded):
    shutil.copy(HERE / "qualification.js", directory)
    shutil.copy(HERE.parent / "workstation_pixels.js", directory)
views = json.loads((root / "docs/manual/images/dicom-metis-real-browser-mri-edge.json").read_text())["views"]
(embedded / "oracle.json").write_text(json.dumps({name: view["rgba_sha256"] for name, view in views.items()}))
