<a id="RITK-DEPS-MOIRAI-09-COEVOLUTION-001"></a>

## RITK-DEPS-MOIRAI-09-COEVOLUTION-001 — Advance ritk past the moirai 0.9 / metis 0.4 provider bumps — blocked
- outcome: `cargo check --workspace --all-targets` resolves and the committed Cargo.lock matches the manifest on a clean checkout.
- evidence 2026-10-14 (manifest requirements read from the local main tip; source cites verified at origin/main): Moirai HEAD is 0.9.0 while 7 moirai edge-sets in ritk's graph require <0.9.0 — ritk direct `^0.8.2`, apollo `^0.8.2` (apollo-fft 0.28.1), Coeus `^0.8.2`, leto `^0.8.2`, consus `^0.8.2` (sibling URL form without `.git`), gaia `>=0.7.0, <0.9.0`, metis 0.4.1 `^0.8.3` — and cargo does not unify a consumer `rev` pin with provider floats (verified: pinning ritk to the moirai 0.8.2 release commit fails identically). No ritk-local manifest edit can fix this. Committed Cargo.lock is fossilized (moirai-runtime 0.7.0, apollo-fft 0.27.0, metis-platform 0.1.0) — advancement commits shipped without lock updates, and RITK-CI-MERGE-GATE-001 (no required check) let them land red.
- acceptance: lock regenerated from the manifest; `cargo check --workspace --all-targets` green; no `rev =` quarantine remains.
- scope: provider order leto, Coeus, apollo, consus (also canonicalize its Moirai URL to the `.git` form), gaia to moirai `^0.9.0`, then ritk (moirai `^0.9.0`, metis `^0.4.1`); regenerate ritk Cargo.lock in the same sweep.
- re-open trigger: apollo, Coeus, leto, consus, and gaia origin/main all require moirai-runtime `^0.9.0`.
- status: blocked
- needs: none
- priority: correctness
