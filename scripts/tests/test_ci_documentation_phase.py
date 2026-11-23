"""Pin the required Rustdoc job to Atlas's bounded documentation phase."""

from __future__ import annotations

import os
import pathlib
import posixpath
import subprocess
import tempfile
import textwrap
import tomllib
import unittest


ROOT = pathlib.Path(
    os.environ.get("RITK_CI_CONTRACT_ROOT", pathlib.Path(__file__).resolve().parents[2])
)
ATLAS_SOURCE = "152264a7df2f6d76feb74cf57155c217dd64a6c7"


class DocumentationPhaseWorkflowTests(unittest.TestCase):
    """Keep Rustdoc ownership complete when the push phase omits it."""

    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "Cargo.toml").open("rb") as stream:
            cls.manifest = tomllib.load(stream)
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        start = workflow.index("\n  rustdoc:\n")
        end = workflow.index("\n  dependency-alignment:\n", start)
        cls.workflow = workflow
        cls.rustdoc = workflow[start:end]

    def test_manifest_assigns_rustdoc_to_pre_merge(self) -> None:
        self.assertEqual(
            self.manifest["workspace"]["metadata"]["atlas"]["documentation-phase"],
            "pre-merge",
        )

    def test_job_uses_pinned_registered_stack_checkout(self) -> None:
        self.assertEqual(
            self.workflow.count(f"ATLAS_DOCUMENTATION_SOURCE: {ATLAS_SOURCE}"), 1
        )
        for required in (
            "repository: ryancinsight/atlas",
            "ref: ${{ env.ATLAS_DOCUMENTATION_SOURCE }}",
            "path: atlas/repos/ritk",
            "fetch-depth: 0",
            'rev-parse "${ATLAS_DOCUMENTATION_SOURCE}:scripts/git-hooks/pre-push"',
            "git hash-object .githooks/pre-push",
        ):
            with self.subTest(required=required):
                self.assertEqual(self.rustdoc.count(required), 1)

    def test_cache_and_config_use_the_canonical_stack_target(self) -> None:
        mapping = "workspaces: atlas/repos/ritk -> ../../target"
        self.assertEqual(self.rustdoc.count(mapping), 1)
        self.assertEqual(
            posixpath.normpath("atlas/repos/ritk/../../target"), "atlas/target"
        )
        preparation = "python3 ../../scripts/atlas-stack-overlay.py off"
        self.assertEqual(self.rustdoc.count(preparation), 1)
        self.assertLess(
            self.rustdoc.index(preparation), self.rustdoc.index("Swatinem/rust-cache")
        )

    def test_job_uses_bounded_phase_without_raw_fallback(self) -> None:
        for required in (
            "python3 -I ../../scripts/pre_push_gate.py",
            "direct documentation .githooks/pre-push",
            'documentation "$CANDIDATE_REVISION" "$BASE_REVISION"',
            'git rev-list --max-parents=0 "$CANDIDATE_REVISION"',
        ):
            with self.subTest(required=required):
                self.assertEqual(self.rustdoc.count(required), 1)
        self.assertEqual(self.rustdoc.count("cargo doc"), 0)

    def test_documentation_range_on_diverged_branch(self) -> None:
        # Execute the exact shell used by the hosted job. Both branch tips
        # advance past their common commit, which itself follows the root.
        step = self.rustdoc.split("id: documentation", 1)[1]
        script = textwrap.dedent(
            step.split("        run: |\n", 1)[1].split("          python3 -I", 1)[0]
        )
        with tempfile.TemporaryDirectory() as directory:
            def git(*args: str) -> str:
                return subprocess.check_output(
                    ["git", "-c", "user.name=CI", "-c", "user.email=ci@example.org", *args],
                    cwd=directory, text=True,
                ).strip()

            def commit(message: str) -> str:
                git("commit", "-q", "--allow-empty", "-m", message)
                return git("rev-parse", "HEAD")

            git("init", "-q", "-b", "main")
            root = commit("root")
            common = commit("common")
            git("branch", "feature")
            main = commit("main")
            git("update-ref", "refs/remotes/origin/main", main)
            git("switch", "-q", "feature")
            tip = commit("feature")
            cases = (
                ("workflow_dispatch", "refs/heads/feature", "", tip, common),
                ("workflow_dispatch", "refs/heads/main", "", main, root),
                ("pull_request", "refs/pull/7/merge", main, tip, main),
                ("push", "refs/heads/main", common, main, common),
            )
            for event, ref, base, candidate, expected in cases:
                with self.subTest(event=event, ref=ref):
                    env = dict(os.environ, GITHUB_EVENT_NAME=event, GITHUB_REF=ref,
                               DEFAULT_BRANCH="main", BASE_REVISION=base,
                               CANDIDATE_REVISION=candidate)
                    actual = subprocess.check_output(
                        ["bash", "-c", script + 'echo "$BASE_REVISION"'],
                        cwd=directory, env=env, text=True,
                    ).strip()
                    self.assertEqual(actual, expected)

    def test_aggregate_binds_success_to_candidate(self) -> None:
        binding = (
            'needs.rustdoc.outputs.documented_candidate }}" != '
            '"${{ github.sha }}'
        )
        self.assertEqual(self.workflow.count(binding), 1)


if __name__ == "__main__":
    unittest.main()
