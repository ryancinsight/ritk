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
        ):
            with self.subTest(required=required):
                self.assertEqual(self.rustdoc.count(required), 1)
        self.assertEqual(self.rustdoc.count("cargo doc"), 0)
        self.assertNotIn("rev-list --max-parents=0", self.rustdoc)
        self.assertRegex(
            self.rustdoc, r"(?m)^ {10}MAIN_DOCUMENTATION_CHECKPOINT: [0-9a-f]{40}$"
        )

    def test_documentation_ranges(self) -> None:
        # Run the workflow's real resolver on a Git history with skipped and
        # failed main changes, a diverged branch, and an actual PR merge commit.
        step = self.rustdoc.split("id: documentation", 1)[1]
        script = textwrap.dedent(
            step.split("        run: |\n", 1)[1].split("          python3 -I", 1)[0]
        ) + 'printf "%s\u005cn" "$BASE_REVISION"'
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            repo = root / "repo"
            repo.mkdir()

            def git(*args: str, cwd: pathlib.Path = repo) -> str:
                return subprocess.check_output(
                    ["git", "-c", "user.name=CI", "-c", "user.email=ci@example.org", *args],
                    cwd=cwd, text=True,
                ).strip()

            def commit(message: str, path: str | None = None) -> str:
                if path is not None:
                    file = repo / path
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_text(message + "\n", encoding="utf-8")
                    git("add", path)
                git("commit", "-q", "--allow-empty", "-m", message)
                return git("rev-parse", "HEAD")

            def resolve(event: str, ref: str, base: str, candidate: str,
                        checkpoint: str, cwd: pathlib.Path = repo) -> subprocess.CompletedProcess:
                return subprocess.run(
                    ["bash", "-c", script], cwd=cwd, text=True, capture_output=True,
                    env=dict(os.environ, GITHUB_EVENT_NAME=event, GITHUB_REF=ref,
                             DEFAULT_BRANCH="main", BASE_REVISION=base,
                             CANDIDATE_REVISION=candidate, MAIN_DOCUMENTATION_CHECKPOINT=checkpoint),
                )

            git("init", "-q", "-b", "main")
            commit("root")
            checkpoint = commit("qualified")
            git("branch", "feature")
            commit("docs-only skipped", "docs/intervening.md")
            failed = commit("failed", "failed.rs")
            main = commit("main", "main.rs")
            git("update-ref", "refs/remotes/origin/main", main)
            git("switch", "-q", "feature")
            tip = commit("topic", "feature.rs")
            git("switch", "-q", "main")
            git("merge", "-q", "--no-ff", "feature", "-m", "synthetic PR merge")
            pr = git("rev-parse", "HEAD")

            for event, ref, given, candidate, expected in (
                ("schedule", "refs/heads/main", "", main, checkpoint),
                ("workflow_dispatch", "refs/heads/main", "", main, checkpoint),
                ("workflow_dispatch", "refs/heads/feature", "", tip, checkpoint),
                ("pull_request", "refs/pull/7/merge", main, pr, main),
                ("push", "refs/heads/main", failed, main, failed),
            ):
                with self.subTest(event=event, ref=ref):
                    git("switch", "-q", "--detach", candidate)
                    result = resolve(event, ref, given, candidate, checkpoint)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip().splitlines()[-1], expected)
                    if event == "schedule" or (event == "workflow_dispatch" and candidate == main):
                        self.assertEqual(
                            git("diff", "--name-only", expected, candidate).splitlines(),
                            ["docs/intervening.md", "failed.rs", "main.rs"],
                        )

            git("switch", "-q", "--detach", pr)
            for pin, diagnostic in (
                ("", "not a full commit SHA"),
                ("0" * 40, "not a local commit"),
                ("1" * 40, "not a local commit"),
                ("HEAD", "not a full commit SHA"),
                (pr, "must differ"),
                (tip, "strict first-parent"),
            ):
                with self.subTest(checkpoint=pin):
                    result = resolve("schedule", "refs/heads/main", "", pr, pin)
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn(diagnostic, result.stderr)

            for event, ref, base, candidate, diagnostic in (
                ("schedule", "refs/heads/feature", "", pr, "must target"),
                ("schedule", "refs/heads/main", "", main, "checkout does not match"),
                ("workflow_dispatch", "refs/tags/version", "", pr, "must target a branch"),
                ("push", "refs/heads/main", "", pr, "not a full commit SHA"),
                ("pull_request", "refs/pull/7/merge", pr, pr, "must differ"),
            ):
                with self.subTest(event=event, ref=ref, base=base):
                    result = resolve(event, ref, base, candidate, checkpoint)
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn(diagnostic, result.stderr)

            git("switch", "-q", "--detach", main)
            unrelated = git("commit-tree", git("rev-parse", "HEAD^{tree}"), "-m", "unrelated")
            result = resolve("push", "refs/heads/main", unrelated, main, checkpoint)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("not an ancestor", result.stderr)

            # Two equally good merge bases: neither can stand in for the other.
            tree = git("rev-parse", "HEAD^{tree}")
            left = git("commit-tree", tree, "-p", tip, "-p", main, "-m", "left")
            right = git("commit-tree", tree, "-p", main, "-p", tip, "-m", "right")
            git("switch", "-q", "--detach", left)
            git("update-ref", "refs/remotes/origin/main", right)
            result = resolve("workflow_dispatch", "refs/heads/feature", "", left, checkpoint)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("no unique main merge base", result.stderr)

            # Both topics have a valid parent, so falling back to candidate^
            # on a merge-base lookup failure would silently accept these ranges.
            unrelated_tip = git("commit-tree", tree, "-p", unrelated, "-m", "unrelated topic")
            git("update-ref", "refs/remotes/origin/main", main)
            for scenario, candidate, has_origin_main in (
                ("unrelated topic history", unrelated_tip, True),
                ("missing origin/main", tip, False),
            ):
                with self.subTest(topic_dispatch=scenario):
                    if not has_origin_main:
                        git("update-ref", "-d", "refs/remotes/origin/main")
                    git("switch", "-q", "--detach", candidate)
                    result = resolve("workflow_dispatch", "refs/heads/feature", "", candidate, checkpoint)
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn("no unique main merge base", result.stderr)

            shallow = root / "shallow"
            git("clone", "-q", "--depth", "1", "--branch", "main",
                repo.as_uri(), str(shallow), cwd=root)
            result = resolve("schedule", "refs/heads/main", "", pr, checkpoint, cwd=shallow)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("requires a complete history", result.stderr)

    def test_aggregate_binds_success_to_candidate(self) -> None:
        binding = (
            'needs.rustdoc.outputs.documented_candidate }}" != '
            '"${{ github.sha }}'
        )
        self.assertEqual(self.workflow.count(binding), 1)


if __name__ == "__main__":
    unittest.main()
