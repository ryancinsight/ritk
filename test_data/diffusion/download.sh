#!/usr/bin/env bash
# Download CC0 diffusion MRI datasets from OpenNeuro for real-data
# integration testing.
#
# OpenNeuro stores large files via git-annex; this script shallow-clones
# the dataset repositories through GitHub mirrors to obtain the text-based
# bval/bvec gradient-scheme files.  The NIfTI volumes need git-annex and are
# not fetched here — the integration tests generate synthetic signals using
# the real gradient schemes, so the bval/bvec pairs are sufficient.
#
# Usage:
#   bash test_data/diffusion/download.sh
#
# Requirements:
#   - git
#
# Datasets:
#   ds002087  — "MRI Datasets with and without deliberate head movements"
#               Single-subject, single-shell (b=700/b=2000), 99 volumes.
#               License: CC0
#               URL: https://openneuro.org/datasets/ds002087
#
#   ds004666  — "EDDEN: Evaluation of Diffusion MRI DENoising"
#               Single-subject, multi-shell (b≈1000/b≈2000), 199 volumes.
#               Three isotropic resolutions (0.9, 1.5, 2.0 mm).
#               License: CC0
#               URL: https://openneuro.org/datasets/ds004666

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

clone_dataset() {
    local id="$1"
    local repo_dir="${SCRIPT_DIR}/${id}_repo"
    local repo_url="https://github.com/OpenNeuroDatasets/${id}.git"

    echo "=== ${id} ==="
    echo "  Source: https://openneuro.org/datasets/${id}"
    echo "  Mirror: ${repo_url}"
    echo "  Dest:   ${repo_dir}"

    if [[ -d "${repo_dir}/.git" ]]; then
        echo "  [skip] Repository already exists."
    else
        echo "  [clone] Shallow-cloning ..."
        git clone --depth 1 "${repo_url}" "${repo_dir}"
        echo "  Clone complete."
    fi
    echo ""
}

echo "================================================"
echo "  RITK Diffusion MRI Test Data Download"
echo "================================================"
echo ""

# ── ds002087 ───────────────────────────────────────────────────────────────
clone_dataset "ds002087"

DWI_087="${SCRIPT_DIR}/ds002087_repo/sub-01/dwi"
if [[ -d "${DWI_087}" ]]; then
    echo "  ds002087 DWI files:"
    ls -lh "${DWI_087}"/*.bval "${DWI_087}"/*.bvec "${DWI_087}"/*.json 2>/dev/null || true
else
    echo "  ERROR: DWI directory not found at ${DWI_087}"
    exit 1
fi
echo ""

# ── ds004666 (EDDEN) ───────────────────────────────────────────────────────
clone_dataset "ds004666"

DWI_666="${SCRIPT_DIR}/ds004666_repo/sub-01/ses-0p9mm/dwi"
if [[ -d "${DWI_666}" ]]; then
    echo "  ds004666 (EDDEN) ses-0p9mm DWI files:"
    ls -lh "${DWI_666}"/*.bval "${DWI_666}"/*.bvec "${DWI_666}"/*.json 2>/dev/null || true
else
    echo "  ERROR: EDDEN DWI directory not found at ${DWI_666}"
    exit 1
fi
echo ""


# ── DWI volumes from OpenNeuro S3 ───────────────────────────────────────────
#
# The DataLad clones above carry git-annex pointer files, not the imaging data:
# a .nii.gz there is a few hundred bytes naming a key. Fetching the content
# normally needs git-annex, which is not a reasonable prerequisite, so the
# volumes come straight from OpenNeuro's public S3 bucket instead.
#
# Size is what distinguishes a real volume from a pointer, so that is what the
# idempotence check tests -- an existence check would accept the pointer.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIN_VOLUME_BYTES=1000000

fetch_volume() {
    local url="$1" dest="$2"
    if [ -f "${dest}" ]; then
        local size
        size=$(wc -c < "${dest}" | tr -d ' ')
        if [ "${size}" -ge "${MIN_VOLUME_BYTES}" ]; then
            echo "  present: $(basename "${dest}") (${size} bytes)"
            return 0
        fi
        echo "  replacing git-annex pointer: $(basename "${dest}")"
    fi
    echo "  fetching $(basename "${dest}") ..."
    if ! curl -fsSL -o "${dest}.partial" "${url}"; then
        echo "  ERROR: download failed for ${url}" >&2
        rm -f "${dest}.partial"
        return 1
    fi
    # Rename only after a complete transfer, so an interrupted run leaves no
    # truncated file that the size check would later accept.
    mv "${dest}.partial" "${dest}"
    echo "  fetched $(basename "${dest}") ($(wc -c < "${dest}" | tr -d ' ') bytes)"
}

echo "ds002087 sub-01 DWI volume (~55 MB):"
fetch_volume     "https://s3.amazonaws.com/openneuro.org/ds002087/sub-01/dwi/sub-01_run-1_dwi.nii.gz"     "${ROOT}/sub-01_dwi.nii.gz"
cp -f "${ROOT}/ds002087_repo/sub-01/dwi/sub-01_run-1_dwi.bval" "${ROOT}/sub-01_dwi.bval" 2>/dev/null || true
cp -f "${ROOT}/ds002087_repo/sub-01/dwi/sub-01_run-1_dwi.bvec" "${ROOT}/sub-01_dwi.bvec" 2>/dev/null || true
cp -f "${ROOT}/sub-01_dwi.nii.gz" "${ROOT}/ds002087_repo/sub-01/dwi/sub-01_run-1_dwi.nii.gz" 2>/dev/null || true
echo ""

# ── Done ────────────────────────────────────────────────────────────────────
echo "================================================"
echo "  Download complete"
echo "================================================"
echo ""
echo "To run the real-data integration tests:"
echo "  cargo test -p ritk-diffusion --test integration_real_data -- --ignored"
echo ""
echo "To regenerate the book figure:"
echo "  cargo run --release -p ritk-diffusion --example book_brain_tractography"
