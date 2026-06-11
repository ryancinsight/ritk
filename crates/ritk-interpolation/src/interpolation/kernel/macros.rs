//! Historical reference for the DRY-353-02 migration.
//!
//! # Status (Sprint 356)
//!
//! **Proc-macro migration complete — `macro_rules!` template removed.**
//! The `interp_dim_template!` `macro_rules!` template that previously
//! lived in this file has been superseded by the proc-macro of the
//! same name in `crates/ritk-macros/` (see audit §7.7). The
//! `DRY_353_02_STATUS` marker is retained below as a historical
//! reference documenting the closure-approach-fails finding.
//!
//! See `docs/audit_optimization_sprint_350.md` §4.2 / §7.7 for the
//! full migration entry.

/// Historical marker for the DRY-353-02 migration status.
///
/// Status: **proc-macro migration complete (Sprint 356) — template removed**.
/// The [`interp_dim_template!`] `macro_rules!` template that previously
/// lived in this file has been superseded by the proc-macro in
/// `crates/ritk-macros/`. This marker is retained as a historical
/// reference documenting the closure-approach-fails finding.
///
/// **Root cause (historical)**: `macro_rules!` introduced a hygiene
/// barrier between identifiers defined inside the macro arm (the
/// prelude's `wz`, `ww`, etc.) and identifiers from the call site
/// (the body). The compiler treated `wz` defined in the prelude and
/// `wz` referenced in the body as different identifiers (different
/// hygiene contexts), so the body failed to compile with "cannot
/// find value `wz` in this scope".
///
/// **Closure-approach-fails finding (Sprint 356)**: the closure-based
/// workaround was investigated and **confirmed to fail**. The
/// closure body is still a token tree from the call site, so
/// identifiers referenced in the body remain in the call-site's
/// hygiene context, while the prelude variables stay in the macro's
/// hygiene context. The same hygiene barrier applies to any macro-arm
/// restructuring attempt (function arguments, struct fields, closure
/// parameters) — the body identifiers never enter the macro's hygiene
/// context.
///
/// **Resolution (Sprint 356)**: rewrite as a procedural macro. A new
/// `ritk-macros` crate (with `proc-macro = true`) was created at
/// `crates/ritk-macros/` and the `interp_dim_template!` proc-macro
/// was implemented. Proc-macros generate code via `TokenStream` and
/// have no hygiene barrier — identifiers defined by the macro and
/// identifiers from the call site are in the same hygiene context
/// after expansion. All 4 `dim{1,2,3,4}.rs` files now use the
/// proc-macro and the build is verified green (64/0/1 interpolation
/// tests pass).
///
/// See `docs/audit_optimization_sprint_350.md` §4.2 / §7.7 for the
/// full migration entry and the closure-approach-fails investigation.
#[allow(dead_code)]
pub const DRY_353_02_STATUS: &str = "proc-macro-migration-complete-template-removed";
