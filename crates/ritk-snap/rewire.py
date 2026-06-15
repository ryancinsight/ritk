"""Rewire inline test blocks to external files via #[path = "..."] mod tests;"""

import os
import re


def rewire(filepath, test_file):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cut = -1
    for i, line in enumerate(lines):
        s = line.rstrip("\n").rstrip("\r")
        # Pattern 1: separator comment like "// -- Tests ----"
        if re.match(r"^// \u2500\u2500 [Tt]ests", s):
            cut = i
            break
        # Pattern 2: top-level #[cfg(test)] followed by mod tests
        if s == "#[cfg(test)]":
            for j in range(i + 1, min(i + 5, len(lines))):
                ns = lines[j].rstrip("\n").rstrip("\r")
                if ns == "" or ns.startswith("//"):
                    continue
                if ns.startswith("mod tests"):
                    cut = i
                break
            if cut >= 0:
                break

    if cut < 0:
        print(f"WARNING: no test block found in {filepath}")
        return

    keep = lines[:cut]
    while keep and keep[-1].strip() == "":
        keep.pop()

    new_content = (
        "".join(keep) + '\n\n#[cfg(test)]\n#[path = "' + test_file + '"]\nmod tests;\n'
    )

    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_content)

    print(f"OK: {filepath} -> {test_file}")


base = r"D:\atlas\repos\ritk\crates\ritk-snap\src"

files = [
    ("render/slice_render.rs", "tests_slice_render.rs"),
    ("render/buffer_pool.rs", "tests_buffer_pool.rs"),
    ("render/colormap.rs", "tests_colormap.rs"),
    ("render/fusion.rs", "tests_fusion.rs"),
    ("render/histogram.rs", "tests_histogram.rs"),
    ("dicom/suv.rs", "tests_suv.rs"),
    ("ui/histogram_interact.rs", "tests_histogram_interact.rs"),
    ("ui/preset_panel.rs", "tests_preset_panel.rs"),
    ("ui/dropped_input.rs", "tests_dropped_input.rs"),
    ("ui/window_level.rs", "tests_window_level.rs"),
    ("ui/mpr_cursor.rs", "tests_mpr_cursor.rs"),
    ("ui/layout.rs", "tests_layout.rs"),
    ("ui/pan.rs", "tests_pan.rs"),
    ("ui/tool_shortcuts.rs", "tests_tool_shortcuts.rs"),
    ("ui/colorbar.rs", "tests_colorbar.rs"),
    ("ui/rtstruct_overlay.rs", "tests_rtstruct_overlay.rs"),
    ("ui/live_preview.rs", "tests_live_preview.rs"),
    ("ui/pointer_intensity.rs", "tests_pointer_intensity.rs"),
    ("ui/histogram.rs", "tests_histogram.rs"),
    ("tools/kind.rs", "tests_kind.rs"),
    ("app/rt_struct_export.rs", "tests/rt_struct_export.rs"),
    ("app/mesh_ops.rs", "tests/mesh_ops.rs"),
    ("app/surface_export.rs", "tests/surface_export.rs"),
    ("lib.rs", "tests_lib.rs"),
]

for rel, test_file in files:
    filepath = os.path.join(base, *rel.split("/"))
    rewire(filepath, test_file)
