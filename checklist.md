# Active targets: descriptor rounding and Python CI

## MIND descriptor contract

- [x] [patch] Reproduce RITK-MIND-HALF-TIE-001 through exact quantizer values, real patches, and public similarity.
- [x] [patch] Restore binary32 half-away rounding and document the contract in ADR0024.
- [x] [patch] Pass all 20 MIND tests and registration Clippy for all targets with warnings denied.
- [x] [patch] Pass 442 registration tests (11 existing skips), workspace formatting, and standalone lock integrity.
- [x] [patch] Pass the Atlas cast gate: 827 registration sites ≤ row 830; no new bare casts.
- [ ] [patch] Review [PR832](https://github.com/ryancinsight/ritk/pull/832), tracked in [RITK-MIND-HALF-TIE-001](backlog/RITK-MIND-HALF-TIE-001.md).

## Python script compatibility and cache budget

- [x] [patch] Reproduce the older-Python lock-parser failure on unchanged main and add regression coverage.
- [x] [patch] Preserve TOML 1.0 validation, selected-interpreter execution and UTF-8/LF gallery evidence bytes.
- [x] [patch] Pass all 41 script tests on Python 3.9, 3.10, 3.11 and 3.13 with unchanged matrix, gates and limits.
- [x] [patch] Measure the cancelled Windows job and remove optional branch cache writes using the pinned action's main-only save policy.
- [x] [patch] Verify PR833 cache post-step timings and the complete Python matrix within the unchanged 30-minute bound in [RITK-PYTHON-CI-CACHE-SAVE-001](backlog/RITK-PYTHON-CI-CACHE-SAVE-001.md).
- [ ] [patch] Review the separate prerequisite and its exact-revision full hosted-matrix evidence in [RITK-PYTHON-SCRIPT-RUNTIME-001](backlog/RITK-PYTHON-SCRIPT-RUNTIME-001.md).
- [ ] [patch] Pass final combined PR-event gates and independent review, then integrate PR833 before PR832.
- [ ] [patch] Resolve [registry publication prerequisites](backlog/RITK-REGISTRY-PUBLICATION-001.md) and verify actual published versions without changing credentials or security settings without specific approval.
