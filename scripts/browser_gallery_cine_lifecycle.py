"""Shared teardown and evidence persistence for the browser cine capture."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

from browser_protocol import ROOT, BrowserRuntimeError, WebDriverClient, _safe_path


STOPPED_STATE_SCRIPT = """
const button = document.getElementById("cine-toggle");
const rate = document.getElementById("cine-rate");
const output = document.getElementById("cine-rate-value");
if (!(button instanceof HTMLButtonElement) ||
    !(rate instanceof HTMLInputElement) ||
    !(output instanceof HTMLOutputElement)) return null;
return {
  button_disabled: button.disabled,
  button_pressed: button.getAttribute("aria-pressed"),
  button_text: button.textContent || "",
  rate_disabled: rate.disabled,
  rate_value: rate.value,
  output: output.textContent || "",
};
"""


def stopped_state(client: WebDriverClient) -> dict[str, Any]:
    """Read and validate the cine controls after viewer teardown."""
    result = client.execute(STOPPED_STATE_SCRIPT)
    if not isinstance(result, dict):
        raise BrowserRuntimeError("stopped cine controls returned an unexpected shape")
    if (
        result.get("button_disabled") is not True
        or result.get("rate_disabled") is not True
        or result.get("button_pressed") != "false"
        or result.get("button_text") != "Play"
        or result.get("rate_value") != "12"
        or result.get("output") != "12 FPS"
    ):
        raise BrowserRuntimeError(f"stopped cine controls retained active state: {result!r}")
    return result


def write_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Persist one bounded cine evidence object at its recorded artifact path."""
    artifact = evidence.get("artifact")
    if not isinstance(artifact, str):
        raise BrowserRuntimeError("cine evidence omitted its artifact path")
    evidence_path = _safe_path(ROOT / artifact, directory=ROOT)
    encoded = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if len(encoded.encode("utf-8")) > 512 * 1024:
        raise BrowserRuntimeError("cine evidence exceeds the 512 KiB trace bound")
    evidence_path.write_text(encoded, encoding="utf-8", newline="\n")
    return evidence


def source_digests(source_root: pathlib.Path) -> dict[str, str]:
    """Hash the scripts that define one cine capture contract."""
    return {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in (
            "browser_gallery_cine.py",
            "browser_gallery_cine_lifecycle.py",
            "browser_gallery.py",
        )
    }


def finalize_cine_teardown(client: WebDriverClient, evidence: dict[str, Any]) -> dict[str, Any]:
    """Complete cine stop evidence after a later control capture owns teardown.

    The combined gallery exercises several RITK control families in one mounted
    lifecycle. A later capture may stop the viewer, so this helper records the
    shared post-stop sample and cine control state in the earlier cine artifact.
    """
    if not isinstance(evidence, dict) or evidence.get("stopped") is not None:
        raise BrowserRuntimeError("cine evidence is not awaiting shared teardown")
    sample_after_stop = client.execute("return window.metisGallery.sample();")
    if (
        not isinstance(sample_after_stop, dict)
        or sample_after_stop.get("mounted") is not False
        or sample_after_stop.get("consumer_listeners") != 0
    ):
        raise BrowserRuntimeError(
            f"RITK viewer did not release listeners after shared stop: {sample_after_stop!r}"
        )
    evidence["samples"]["after_stop"] = sample_after_stop
    evidence["stopped"] = stopped_state(client)
    evidence["teardown"] = {"performed": True, "owner": "tool-controls"}
    return write_evidence(evidence)
