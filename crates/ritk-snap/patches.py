"""Apply P33-P38 source-code patches."""

import os
import re

BASE = r"D:\atlas\repos\ritk\crates\ritk-snap\src"


def path(*parts):
    return os.path.join(BASE, *parts)


def read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, content):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"wrote {p}")


def add_import(content, import_line, after_pattern):
    """Insert import_line after the line matching after_pattern."""
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if after_pattern in line:
            lines.insert(i + 1, import_line)
            return "\n".join(lines)
    # Fallback: insert at top (after any shebang/doc attrs)
    lines.insert(0, import_line)
    return "\n".join(lines)


# ── P33: DEFAULT_WINDOW_CENTER / DEFAULT_WINDOW_WIDTH ─────────────────────────

P33_FILES = [
    "app/render_cache.rs",
    "app/io_ops.rs",
    "app/panels.rs",
    "app/pointer_ops.rs",
    "app/viewport_render.rs",
    "app/viewport_compare.rs",
    "app/toolbar.rs",
]

P33_IMPORT = "use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};"

for rel in P33_FILES:
    p = path(*rel.split("/"))
    content = read(p)

    # Skip if already imported
    if "DEFAULT_WINDOW_CENTER" in content:
        print(f"SKIP (already patched): {p}")
        continue

    # Add import after the first `use super::state` line
    content = add_import(content, P33_IMPORT, "use super::state")

    # Replace the literal values
    content = content.replace("unwrap_or(128.0)", "unwrap_or(DEFAULT_WINDOW_CENTER)")
    content = content.replace("unwrap_or(256.0)", "unwrap_or(DEFAULT_WINDOW_WIDTH)")

    write(p, content)


# ── P34: MPR info-panel height constants in viewport.rs ───────────────────────

VIEWPORT = path("app", "viewport.rs")
content = read(VIEWPORT)

if "MPR_INFO_HEIGHT_FRAC" not in content:
    # Insert constants after the opening use statement
    consts = (
        "\n"
        "// ── Layout constants ─────────────────────────────────────────────────────────\n"
        "\n"
        "/// Fraction of the available height reserved for the MPR info panel.\n"
        "const MPR_INFO_HEIGHT_FRAC: f32 = 0.24;\n"
        "\n"
        "/// Minimum pixel height of the MPR info panel.\n"
        "const MPR_INFO_MIN_H: f32 = 110.0;\n"
        "\n"
        "/// Maximum pixel height of the MPR info panel.\n"
        "const MPR_INFO_MAX_H: f32 = 210.0;\n"
    )
    # Insert after the last `use` line before `impl`
    content = re.sub(
        r"(use crate::ui::AnatomicalPlane;)",
        r"\1" + consts,
        content,
    )
    # Replace magic numbers in clamp expressions
    content = content.replace("avail.y * 0.24", "avail.y * MPR_INFO_HEIGHT_FRAC")
    content = content.replace(
        ".clamp(110.0, 210.0)", ".clamp(MPR_INFO_MIN_H, MPR_INFO_MAX_H)"
    )
    write(VIEWPORT, content)

# ── P34: Overlay label constants in viewport_render.rs ────────────────────────
#   6.0 (inset), 12.0 (font), (255,255,255,210) (colour) appear in 3 files.

VP_RENDER = path("app", "viewport_render.rs")
content = read(VP_RENDER)

if "OVERLAY_LABEL_INSET" not in content:
    overlay_consts = (
        "\n"
        "// ── Overlay label constants ──────────────────────────────────────────────────\n"
        "\n"
        "/// Pixel inset from the viewport corner for overlay text labels.\n"
        "pub(crate) const OVERLAY_LABEL_INSET: f32 = 6.0;\n"
        "\n"
        "/// Font size for viewport overlay text labels (proportional points).\n"
        "pub(crate) const OVERLAY_LABEL_FONT_SIZE: f32 = 12.0;\n"
        "\n"
        "/// Colour for viewport overlay text labels (white @ 82 % opacity, premultiplied).\n"
        "pub(crate) const OVERLAY_LABEL_COLOR: egui::Color32 =\n"
        "    egui::Color32::from_rgba_premultiplied(210, 210, 210, 210);\n"
    )
    # Insert after the last top-level `use` line
    content = re.sub(
        r"(use crate::ui::overlay::\{OverlayContext, OverlayRenderer\};)",
        r"\1" + overlay_consts,
        content,
    )
    # Replace occurrences in this file
    content = content.replace(
        "egui::vec2(6.0, 6.0)",
        "egui::vec2(OVERLAY_LABEL_INSET, OVERLAY_LABEL_INSET)",
    )
    content = content.replace(
        "egui::FontId::proportional(12.0)",
        "egui::FontId::proportional(OVERLAY_LABEL_FONT_SIZE)",
    )
    content = content.replace(
        "egui::Color32::from_rgba_unmultiplied(255, 255, 255, 210)",
        "OVERLAY_LABEL_COLOR",
    )
    write(VP_RENDER, content)

# Patch render_cache.rs
RCACHE = path("app", "render_cache.rs")
content = read(RCACHE)
if "OVERLAY_LABEL_INSET" not in content:
    content = add_import(
        content,
        "use super::viewport_render::{OVERLAY_LABEL_COLOR, OVERLAY_LABEL_FONT_SIZE, OVERLAY_LABEL_INSET};",
        "use super::state::ProjectionMode;",
    )
    content = content.replace(
        "egui::vec2(6.0, 6.0)",
        "egui::vec2(OVERLAY_LABEL_INSET, OVERLAY_LABEL_INSET)",
    )
    content = content.replace(
        "egui::FontId::proportional(12.0)",
        "egui::FontId::proportional(OVERLAY_LABEL_FONT_SIZE)",
    )
    content = content.replace(
        "egui::Color32::from_rgba_unmultiplied(255, 255, 255, 210)",
        "OVERLAY_LABEL_COLOR",
    )
    write(RCACHE, content)

# Patch viewport_compare.rs
VPCMP = path("app", "viewport_compare.rs")
content = read(VPCMP)
if "OVERLAY_LABEL_INSET" not in content:
    content = add_import(
        content,
        "use super::viewport_render::{OVERLAY_LABEL_COLOR, OVERLAY_LABEL_FONT_SIZE, OVERLAY_LABEL_INSET};",
        "use super::state::SnapApp;",
    )
    content = content.replace(
        "egui::vec2(6.0, 6.0)",
        "egui::vec2(OVERLAY_LABEL_INSET, OVERLAY_LABEL_INSET)",
    )
    content = content.replace(
        "egui::FontId::proportional(12.0)",
        "egui::FontId::proportional(OVERLAY_LABEL_FONT_SIZE)",
    )
    content = content.replace(
        "egui::Color32::from_rgba_unmultiplied(255, 255, 255, 210)",
        "OVERLAY_LABEL_COLOR",
    )
    write(VPCMP, content)


# ── P35: DEFAULT_VR_ALPHA in render_cache.rs ─────────────────────────────────

RCACHE = path("app", "render_cache.rs")
content = read(RCACHE)
if "DEFAULT_VR_ALPHA" not in content:
    content = re.sub(
        r"(use super::state::ProjectionMode;)",
        r"\1\n\n/// Per-voxel opacity scale for volume rendering. Canonical value per GPU VR spec.\nconst DEFAULT_VR_ALPHA: f32 = 0.06;",
        content,
    )
    content = content.replace(", 0.06)", ", DEFAULT_VR_ALPHA)")
    write(RCACHE, content)


# ── P36: DEFAULT_FUSION_ALPHA in app/state.rs and app/volume_state.rs ────────

for rel in ["app/state.rs", "app/volume_state.rs"]:
    p = path(*rel.split("/"))
    content = read(p)
    if (
        "DEFAULT_FUSION_ALPHA" not in content
        and "compare_fusion_alpha: 0.35" in content
    ):
        # Add const near top after use statements
        content = re.sub(
            r"(// ── Helper types|pub\(crate\) type)",
            r"/// Default opacity for the fused-overlay compare mode.\npub(crate) const DEFAULT_FUSION_ALPHA: f32 = 0.35;\n\n\1",
            content,
            count=1,
        )
        content = content.replace(
            "compare_fusion_alpha: 0.35", "compare_fusion_alpha: DEFAULT_FUSION_ALPHA"
        )
        write(p, content)


# ── P37: Rename dot3/cross3 in render/mesh_render.rs + tests ─────────────────

MESH_RENDER = path("render", "mesh_render.rs")
content = read(MESH_RENDER)
if "pub(crate) fn dot3" in content:
    content = content.replace("pub(crate) fn dot3", "pub(crate) fn dot")
    content = content.replace("pub(crate) fn cross3", "pub(crate) fn cross")
    # Replace call sites (dot3( -> dot(, cross3( -> cross()
    content = re.sub(r"\bdot3\(", "dot(", content)
    content = re.sub(r"\bcross3\(", "cross(", content)
    write(MESH_RENDER, content)

MESH_TESTS = path("render", "tests_mesh_render.rs")
content = read(MESH_TESTS)
if "dot3" in content or "cross3" in content:
    content = re.sub(r"\bdot3\b", "dot", content)
    content = re.sub(r"\bcross3\b", "cross", content)
    write(MESH_TESTS, content)


# ── P38: Rename cross3/normalize3 in anatomical_plane.rs ─────────────────────

ANAT = path("ui", "anatomical_plane.rs")
content = read(ANAT)
if "fn cross3" in content or "fn normalize3" in content:
    content = content.replace("fn cross3(", "fn cross(")
    content = content.replace("fn normalize3(", "fn normalize(")
    content = re.sub(r"\bcross3\(", "cross(", content)
    content = re.sub(r"\bnormalize3\(", "normalize(", content)
    write(ANAT, content)


# ── P38: Rename normalize3 in render/gpu_mesh/mesh_buf.rs ────────────────────

MESH_BUF = path("render", "gpu_mesh", "mesh_buf.rs")
content = read(MESH_BUF)
if "fn normalize3" in content:
    content = content.replace("fn normalize3(", "fn normalize(")
    content = re.sub(r"\bnormalize3\(", "normalize(", content)
    write(MESH_BUF, content)

print("P33-P38 done.")
