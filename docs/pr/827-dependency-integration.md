# RITK #827 dependency integration

This in-flight review checklist follows the repository's per-item backlog and
Atlas's `docs/pr/` review-artifact convention; completed root checklist boards
are retired by the board conformance gate.

Target versions: existing RITK packages; Consus 0.2.0, Moirai 0.8.2,
Coeus 0.12.0 and Metis 0.3.2. Phase: Closure of the dependency lock repair.
Release status: blocked; PR #827 remains a draft.

- [x] [patch] Preserve the seven original commits and Cargo.toml contents.
- [x] [patch] Verify consumed Consus/Metis/Moirai APIs and upstream requirements.
- [x] [patch] Reconcile against current main and the separate Metis repair PR.
- [x] [patch] Regenerate Cargo.lock with canonical Git package identities.
- [x] [patch] Verify locked all-feature metadata, lock integrity and provider identity.
- [x] [patch] Verify dependency alignment, formatting and 37 Python script tests.
- [x] [patch] Run Clippy, nextest, rustdoc and keep-going checks; record failures.
- [x] [patch] Record the separate Consus standalone resolver incompatibility.
- [ ] [major] Approve the Coeus public error-contract ADR before implementation.
- [ ] [major] Migrate every affected caller and validate computed values.
- [ ] [patch] Pass the complete Rust, Python wheel and parity CI gates.
