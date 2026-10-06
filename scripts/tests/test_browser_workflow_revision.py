"""Execute the SNAP workflow's lock-derived provider selection."""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import sys
import tempfile
import textwrap
import unittest

_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root.parent / "metis" / "scripts"))
from process_tree import run


class BrowserWorkflowRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = (_root / ".github/workflows/metis-browser-dicom.yml").read_text(
            encoding="utf-8"
        )
        step = re.search(
            r"(?ms)^      - name: Read lock-pinned Métis revision\n"
            r"(?P<step>.*?)^      - name:", cls.workflow
        )
        if step is None:
            raise AssertionError("the workflow must resolve Métis before checkout")
        cls.step = step.group("step")
        cls.script = textwrap.dedent(cls.step.split("        run: |\n", 1)[1])
        confirmation = re.search(
            r"(?ms)^      - name: Confirm the framework revision is locked\n"
            r"(?P<step>.*?)^      - name:", cls.workflow
        )
        if confirmation is None:
            raise AssertionError("the workflow must verify the provider checkout")
        cls.confirmation = textwrap.dedent(
            confirmation.group("step").split("        run: |\n", 1)[1]
        )
        cls.bash = shutil.which("bash")
        if cls.bash is None:
            raise AssertionError("the workflow shell is required")

    def resolve(self, revisions, override="", sources=None):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "ritk").mkdir()
            if sources is None:
                sources = [
                    f"git+https://github.com/ryancinsight/metis.git#{revision}"
                    for revision in revisions
                ]
            (root / "ritk/Cargo.lock").write_text(
                "\n".join(
                    f'[[package]]\nname = "metis-fixture-{index}"\n'
                    f'version = "0.1.0"\n'
                    f'source = "{source}"\n'
                    for index, source in enumerate(sources)
                ),
                encoding="utf-8",
            )
            output = root / "output"
            output.touch()
            environment = dict(os.environ)
            environment.update(
                GITHUB_OUTPUT=str(output), METIS_REVISION_OVERRIDE=override,
                PATH=str(pathlib.Path(self.bash).parent) + os.pathsep + environment["PATH"],
            )
            result = run(
                [self.bash, "-c", self.script], cwd=root, env=environment, timeout=10
            )
            return result, output.read_text(encoding="utf-8")

    def test_lock_advance_alone_changes_selected_revision(self):
        for revision in ("1" * 40, "2" * 40):
            with self.subTest(revision=revision):
                result, output = self.resolve([revision] * 6)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(output, f"revision={revision}\n")

    def test_matching_dispatch_override_uses_locked_revision(self):
        revision = "a" * 40
        result, output = self.resolve([revision], override=revision)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, f"revision={revision}\n")

    def test_mismatched_or_nonimmutable_override_fails_before_output(self):
        for override in ("b" * 40, "main", "a" * 39, "$(exit 0)"):
            with self.subTest(override=override):
                result, output = self.resolve(["a" * 40], override=override)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output, "")
                self.assertIn("override must match Cargo.lock", result.stderr)

    def test_absent_conflicting_or_malformed_lock_revision_fails(self):
        for revisions in ([], ["a" * 40, "b" * 40], ["main"], [""],
                          ["a" * 40, "b" * 39], ["A" * 40]):
            with self.subTest(revisions=revisions):
                result, output = self.resolve(revisions)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output, "")

    def test_every_provider_source_form_is_validated(self):
        provider = "git+https://github.com/ryancinsight/metis"
        locked = provider + ".git#" + "a" * 40
        for source in (provider + ".git?rev=release#" + "b" * 40,
                       provider + ".git", provider + "#main"):
            with self.subTest(source=source):
                result, output = self.resolve([], sources=[locked, source])
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output, "")
        result, output = self.resolve([], sources=[
            locked, provider + "?branch=main#" + "a" * 40,
            "git+https://github.com/ryancinsight/other.git#" + "b" * 40,
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, "revision=" + "a" * 40 + "\n")

    def test_checkout_confirmation_uses_revision_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for command in (
                ["git", "init", "metis"],
                ["git", "-C", "metis", "-c", "user.name=Fixture",
                 "-c", "user.email=fixture@example.org", "commit", "--allow-empty",
                 "-m", "Fixture"],
            ):
                result = run(command, cwd=root, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
            head = run(["git", "-C", "metis", "rev-parse", "HEAD"], cwd=root, timeout=10)
            self.assertEqual(head.returncode, 0, head.stderr)
            revision = head.stdout.strip()
            for suffix in (".git#", ".git?rev=release#", "#"):
                source = "git+https://github.com/ryancinsight/metis" + suffix + revision
                result, output = self.resolve([], sources=[source])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(output, f"revision={revision}\n")
                environment = dict(os.environ, METIS_REVISION=revision)
                result = run([self.bash, "-c", self.confirmation], cwd=root,
                             env=environment, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
            environment["METIS_REVISION"] = "0" * 40
            result = run([self.bash, "-c", self.confirmation], cwd=root,
                         env=environment, timeout=10)
            self.assertNotEqual(result.returncode, 0)

    def test_build_and_browser_checkouts_share_resolved_output(self):
        checkouts = re.findall(
            r"(?ms)^          repository: ryancinsight/metis\n"
            r"          ref: (.*?)\n", self.workflow
        )
        self.assertEqual(checkouts, [
            "${{ steps.metis-revision.outputs.revision }}",
            "${{ needs.build-browser.outputs.metis-revision }}",
        ])
        self.assertRegex(self.workflow, r"(?m)^      metis-revision: "
                         r"\$\{\{ steps.metis-revision.outputs.revision \}\}$")
        self.assertRegex(self.workflow, r"(?m)^    needs: build-browser$")
        self.assertIn("METIS_REVISION_OVERRIDE: ${{ inputs.metis_revision }}", self.step)
        dispatch = self.workflow.split("      metis_revision:\n", 1)[1].split("\n\n", 1)[0]
        self.assertIn("        required: false", dispatch)
        self.assertNotRegex(dispatch, r"(?m)^        default:")
        self.assertNotRegex(self.workflow, r"(?m)^  METIS_REVISION:")


if __name__ == "__main__":
    unittest.main()
