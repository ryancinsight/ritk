# RITK execution backlog

<a id="RITK-PYTHON-FREETHREADED-001"></a>
## RITK-PYTHON-FREETHREADED-001 — Ship free-threaded Python bindings [arch]
- Status: in-progress; priority: P1; owner: RITK Python; integrator: root; branch: `fix/ritk-free-threaded-toolchain`; last-update: 2026-09-21; delivery: PR [#570](https://github.com/ryancinsight/ritk/pull/570), merge `6693d9b9821caeb03d01b5fa58a48fcac2abecc9`; follow-up fixes the merged-main toolchain-component red run.
- Outcome: RITK exposes a real PyO3 free-threaded module contract while retaining the Python 3.9 stable-ABI wheel.
- Scope: PyO3 0.29 migration, `gil_used = false`, `abi3t` feature, concurrent binding test, release and CI matrix, README and ADR 0049. Zero-copy changes and DICOM domain logic remain out of scope.
- Acceptance: `cargo check -p ritk-python --all-targets`, strict Clippy, and nextest pass locally; the hosted CPython 3.15t `abi3t` wheel test must import with the GIL disabled and preserve exact shape, metadata and pixels under concurrent `Image` reads.
- Dependency: PyO3 0.29.2 and Atlas reusable wheel workflow support for `abi3t`/3.15t.
- Verification: local `cargo check` (default and `abi3t`), strict Clippy, formatting, and nextest pass; Python 3.13 skips the free-threaded test by its declared interpreter guard. Hosted 3.15t wheel verification remains the merge gate.
- Failure basis: merged-main runs `35645202445` and `35645201560` exposed PyO3 test-link failures from `extension-module` and an unavailable exact `3.15t` setup. Run `35657075502` then reached CPython 3.15t but failed while resolving VTK, which has no compatible prerelease free-threaded wheel; PR #570 split the dependency set. Run `35658218463` reached the wheel build but failed because the job omitted the `rustfmt` and `clippy` components required by `rust-toolchain.toml`; this increment aligns the setup action with that declaration.

<a id="RITK-CONFORMANCE-001"></a>
## RITK-CONFORMANCE-001 — Restore stack conformance bounds [patch]
- Status: done; priority: P0; owner: RITK integration; integrator: root/ritk_consumer; last-update: 2026-09-21; delivery: PR [#562](https://github.com/ryancinsight/ritk/pull/562), merge `2e346c0dd29d6f711e387167f43df029437eccd2`.
- Outcome: the stack pointer accepts RITK without increasing its recorded source, assertion, or image-debt baselines.
- Scope: split four oversized `ritk-snap` modules by existing concerns, replace two existence-only selection assertions with value checks, and reduce four inspected manual PNGs below the 200 KiB asset budget while preserving their documented content.
- Acceptance: `oversized_files` returns 49→45, `existence_only_assertions` 2→0, and `oversized_tracked_images` 30→26; focused `ritk-snap` formatting, strict Clippy, tests, and documentation gates pass.
- Verification: staged-tree conformance reports zero regressions; native nextest passes 530/530; native and wasm32 strict Clippy, formatting, four doctests with one intentional ignore, and warning-clean Rustdoc pass. The four lossless WebP figures retain the PNG dimensions and decoded pixels exactly.
- Delivery evidence: the merged revision preserves the exact conformance counts and the hosted PR checks passed the artifact budget, lockfile, formatting, Clippy, Rustdoc, dependency-alignment and SemVer checks; the remaining platform test matrix is independent of this source and image-budget closure.

<a id="RITK-JPEG-001"></a>
## RITK-JPEG-001 — Consume shared JPEG raster codec [arch] [major]
- Status: done; integrator: root/ritk_consumer; delivery: [PR 556](https://github.com/ryancinsight/ritk/pull/556), merge `fc85dad03a6c14a617e9687609044497c1eba122`, and [PR 561](https://github.com/ryancinsight/ritk/pull/561), merge `71b247c0f0e948614a7e2c5205b34106762c0861`; updated: 2026-09-21.
- Driver: [METIS-ASSETS-001](../metis/backlog.md#METIS-ASSETS-001).
- Outcome: JPEG byte parsing, entropy reconstruction and EXIF interpretation move to [Consus](../consus/backlog.md#CONSUS-RASTER-001); RITK retains DICOM pixel layout, signedness, modality conversion and clinical presentation.
- Scope: `ritk-codecs` JPEG, `ritk-jpeg` readers, PNG adapters and Snap PNG output, manifests, tests and owning documentation.
- Acceptance: preserve lossless and modality samples, RGB/grayscale file semantics and writer boundaries while deleting duplicate codec computation.
- Dependencies: [Consus PR 80](https://github.com/ryancinsight/consus/pull/80) and [Apollo PR 526](https://github.com/ryancinsight/apollo/pull/526) are merged; the standalone lock resolves both providers coherently.
- Verification: the standalone lock resolves Consus's merged raster provider and Apollo's merged DCT provider coherently; BitsStored propagation, unsigned-scalar cardinality and shared display mapping pass 1,356 debug and 392 release tests, strict Clippy, ten doctests and warning-denied Rustdoc, with five doctests intentionally ignored. ADR 0048 records the required major API migration; release versioning remains a separate authorized action.
- Prior delivery verification (PR 556): standalone lock check passes with 62 first-party Git sources; the five-package debug gate passes 1,299/1,299 and the release codec gate passes 339/339; direct RGB uses the derived one-code-value differential bound; PNG readers reject JPEG bytes under both extensions and Snap emits a PNG signature for a `.jpg` path; the configured pre-push gate passes.
- Prior delivery evidence (PR 556): the post-merge real-study replay reads 94 files/49,807,236 bytes, exits 0, rejects the invalid-study probe with exit 1, and reproduces the committed 1280×800 MRI PNG byte-identically (`259dd791...`, 411,589 non-black pixels) under lock `b1d7c99a6dcbf788f515f4b18789d10035805e38db10f4a32dd7970b3050e226`.
<a id="RITK-METIS-LOCK-015"></a>
## RITK-METIS-LOCK-015 — Replay the merged Métis semantic capture provider [patch]
- Status: done; priority: P1; owner: RITK viewer + integration; integrator: root; last-update: 2026-09-21.
- Outcome: the standalone RITK lock and browser workflow resolve the six Métis packages at merged provider revision `6e53c8a082447e6cd3b4ed46b2173b071b0c37f8`, while the real 94-file MRI replay remains byte-identical and DICOM ownership stays in RITK.
- Scope: first-party Cargo.lock resolution, browser workflow default revision, current real-study provenance and manual synchronization. Semantic-tree capture is host-neutral Métis evidence; DICOM discovery, decoding, geometry and clinical semantics remain RITK-owned.
- Acceptance: standalone Cargo.lock resolves without the Atlas overlay; locked native/WASM checks, strict Clippy, formatting, rustdoc, provenance and real-study replay pass; the 1280×800 PNG remains byte-identical (`259dd791...`, 411,589 non-black pixels); the browser workflow checks out the same full revision by default.
- Dependencies: Métis PR #311 merge `6e53c8a082447e6cd3b4ed46b2173b071b0c37f8`; current Moirai lock source remains unchanged unless Cargo resolution requires its existing revision.
- Verification: standalone lock check, native/WASM gates, strict Clippy, formatting, rustdoc, provenance and the 94-file replay passed on the delivery revision.
- Delivery: PR [#555](https://github.com/ryancinsight/ritk/pull/555), merge `d46fbda252b06a1445272167bdade4403e7bcb49`; the later JPEG integration in PR #556 advances the same six package sources to `4bceb90fe616465eca91cddc0c182548b295ca95`.
<a id="RITK-METIS-LOCK-017"></a>
## RITK-METIS-LOCK-017 - Consume the merged Metis editable accessibility surface [patch]
- Status: done; priority: P1; owner: RITK viewer + integration; integrator: root; last-update: 2026-09-21; dependency: Metis PR #317.
- Outcome: the standalone RITK lock and browser workflow consume Metis merge `3a3b30b0c3481db00b16175e6418907a6ae6f11c` while DICOM decoding, clinical presentation and the byte-identical real MRI replay remain RITK-owned.
- Scope: six Metis source revisions in `Cargo.lock`, the browser workflow default revision, current replay provenance and manual synchronization. No DICOM parser or viewer semantics move into Metis.
- Acceptance: standalone lock resolves without the Atlas overlay; locked native/WASM tests, strict Clippy, formatting, rustdoc, provenance and the real 94-file MRI replay pass; the 1280x800 PNG remains byte-identical (`259dd791...`, 411,589 non-black pixels); the browser workflow checks out the merged Metis revision.
- Verification: the standalone lock check reports 62 first-party Git sources at SHA-256 `ce73c2f...`; `ritk-snap` nextest passes 491/491; native strict Clippy, WASM check/Clippy, formatting, rustdoc and doctests (4 passed, 1 ignored) pass; projection tests pass 5/5; Python script tests pass 26/26; the native replay reads 94 files/49,807,236 bytes, exits 0, rejects the invalid-study probe with exit 1, and preserves the 1280x800 image (`259dd791...`, 411,589 non-black pixels).
- Delivery: PR [#559](https://github.com/ryancinsight/ritk/pull/559), merge
  `fcb6dde73c9c1e55cac959586551dc0f4ed7c6ee`; DICOM remains RITK-owned and the
  provider lock is standalone outside the Atlas overlay.

<a id="RITK-METIS-LOCK-016"></a>
## RITK-METIS-LOCK-016 — Replay the landed Métis accessibility provider [patch]
- Status: done; priority: P1; owner: RITK viewer + integration; integrator: root; last-update: 2026-09-21; dependencies: Metis PR #316.
- Outcome: the standalone lock and browser workflow consume Metis `d0585dac60fa66ae226e7b0b3da6081d834b8f47` while the real 94-file MRI replay remains byte-identical and DICOM ownership stays in RITK.
- Scope: first-party Cargo.lock resolution, browser workflow default revision, replay provenance and manual synchronization; no DICOM parser, viewer semantics or Metis implementation enters RITK through this lock advance.
- Acceptance: standalone lock resolves without the Atlas overlay; locked native/WASM checks, strict Clippy, formatting, rustdoc, provenance and real-study replay pass; the 1280×800 PNG remains byte-identical (`259dd791...`, 411,589 non-black pixels).
- Verification: standalone lock check reports 62 first-party Git sources; `ritk-snap` nextest passes 491/491; native strict Clippy, WASM check/Clippy, rustdoc, doctests (4 passed, 1 ignored), formatting and Python script tests (26/26) pass; the native replay reads 94 files/49,807,236 bytes, exits 0, rejects the invalid-study probe with exit 1, and preserves the 1280×800 image.
- Delivery: PR [#558](https://github.com/ryancinsight/ritk/pull/558), source commit `e8c674b08f460659d00e68378b1e4dde4963b41a`; lock SHA `363419d6f6bd632151aacb135ad847ba84e637e69c83e2ea048e74dfa383101c`; the typed accessibility request path is bounded and reports an explicit unsupported action until RITK installs a native accessibility semantics tree.
<a id="RITK-SNAP-METIS-CROSSHAIR-001"></a>
## RITK-SNAP-METIS-CROSSHAIR-001 — Present linked MPR cursor through Métis hosts [arch] [minor]
- Status: done; priority: P1; owner: RITK presentation; integrator: root; last-update: 2026-09-20.
- Outcome: the host-neutral viewer snapshot carries the linked voxel cursor and visibility state; the Métis browser gallery and Windows native session render the same crosshair over the real RITK planes without moving DICOM or cursor semantics into Métis.
- Scope: `ritk-snap` presentation snapshot, browser semantic attributes and gallery control, native Métis display-list overlay, tests, ADR and manual evidence. Measurement/RT overlays, OS accessibility bridges and WebGPU fallback remain separate items.
- Acceptance: toggling the browser control changes only the cursor overlay state; all three canvases publish identical visibility and cursor coordinates; native and browser projection use the shared `[z,y,x]` cursor mapping under anisotropic spacing and orientation; crosshair pixels/overlay geometry are value-tested, real MRI presentation remains non-black, and locked native/WASM, strict Clippy, rustdoc, formatting and browser script checks pass.
- Dependencies: existing `PresentationSnapshot` contract, `LinkedCursor` mapping, Métis `DisplayCommand::DrawLine`, and the current 94-file MRI replay.
- Delivery: RITK PR [#547](https://github.com/ryancinsight/ritk/pull/547), merge `7eb4a0a1513248755d7d09b7ac8e3363113163a7`; ADR correction PR [#548](https://github.com/ryancinsight/ritk/pull/548), merge `cfe61bd271b47c1bfaf8694bb7f136512eeb3180`.
- Verification: locked native `ritk-snap` nextest 488/488, strict native Clippy, wasm32 check/Clippy, rustdoc, doctests 4/4, release build, formatting, lockfile, Python 24/24, browser JavaScript syntax and workflow attribute checks pass. Native display-list tests assert six crosshair lines, hidden-state removal and orientation-aware movement; browser trace and gallery tests validate the optional cursor group and three-canvas linked state. Hosted run [35530079967](https://github.com/ryancinsight/ritk/actions/runs/35530079967) passes the Chromium window/projection and Firefox raster lanes; artifact [10611440694](https://github.com/ryancinsight/ritk/actions/runs/35530079967/artifacts/10611440694) records `false -> true -> false`, cursor `46,255,255`, and visible overlays on all three real MRI planes. WebKit bounded read and Chromium WebGPU adapter remain separate residuals.

<a id="RITK-SNAP-METIS-ANNOTATIONS-001"></a>
## RITK-SNAP-METIS-ANNOTATIONS-001 — Publish completed browser annotation results [arch] [minor]
- Status: done; priority: P1; owner: RITK presentation; integrator: root; last-update: 2026-09-20.
- Outcome: the host-neutral snapshot and Métis browser canvases publish the completed annotation count, kind and primary value after real Length, Angle, ROI and HU gestures.
- Scope: `PresentationSnapshot`, browser semantic attributes, browser trace/value tests, manual and ADR. Annotation math, DICOM decoding, native drawing and WebKit/WebGPU capability residuals remain separate.
- Acceptance: completed browser gestures publish input-sensitive annotation results with finite values; invalid selection probes leave the result unchanged; all three canvases agree; locked native/WASM, strict Clippy, rustdoc, formatting, browser script and real-study visual checks pass.
- Dependencies: existing `Annotation` calculations, `PresentationSnapshot`, Métis semantic attributes and the 94-file MRI browser replay.
- Delivery: RITK PR [#550](https://github.com/ryancinsight/ritk/pull/550), merge `2a2cc592fa3148eba12ac126da82afc3cdd56b30`; host-gating fix PR [#551](https://github.com/ryancinsight/ritk/pull/551), merge `724091c09fae2b0ce0442094c663721a47732e68`; replay-order fix PR [#552](https://github.com/ryancinsight/ritk/pull/552), merge `651b4a808374afc3efca35b71b9583b0a2a17248`.
- Verification: merged-main hosted run [35535831906](https://github.com/ryancinsight/ritk/actions/runs/35535831906) passes Chromium raster, Chromium window, Chromium MIP projection and Firefox on the 94-file public MRI-DIR study. The Chromium-window artifact [10613092926](https://github.com/ryancinsight/ritk/actions/runs/35535831906/artifacts/10613092926) records Length `102.75155`, Angle `3.5569937`, ROI Rect `5402.25`, ROI Ellipse `4146.0703` and HU Point `27.0`, seven rejected invalid probes, and listener cleanup from 21/34 mounted to 0/0 stopped. The Chromium raster artifact [10613022631](https://github.com/ryancinsight/ritk/actions/runs/35535831906/artifacts/10613022631), Firefox artifact [10612608212](https://github.com/ryancinsight/ritk/actions/runs/35535831906/artifacts/10612608212) and projection artifact [10612576819](https://github.com/ryancinsight/ritk/actions/runs/35535831906/artifacts/10612576819) bind exact RGBA/semantic evidence; WebKit artifact [10612972422](https://github.com/ryancinsight/ritk/actions/runs/35535831906/artifacts/10612972422) and WebGPU artifact [10613591260](https://github.com/ryancinsight/ritk/actions/runs/35535831906/artifacts/10613591260) remain explicit host-capability residuals.
- Documentation delivery: commit `ee49b3905`; PR [#553](https://github.com/ryancinsight/ritk/pull/553) binds the manual, ADR 0047, actual browser captures, and current provenance JSON to the merged-main run.
<a id="RITK-SNAP-NATIVE-PROJECTION-REUSE-001"></a>
## RITK-SNAP-NATIVE-PROJECTION-REUSE-001 — Reuse native scalar projection storage [patch]
- Status: done; priority: P1; owner: RITK native presentation; integrator: root; last-update: 2026-09-20.
- Outcome: repeated native Métis scalar projection refreshes reuse caller-owned scalar and RGBA storage while preserving the existing MIP, MinIP and Average pixels, dimensions, spacing and labels.
- Scope: native scalar projection presentation, reusable projection scratch, value-semantic differential tests, native manual/resource evidence. DICOM decoding, slab semantics, browser/WebGPU presentation and host ownership remain unchanged.
- Acceptance: the initial and subsequent scalar projection renders are byte-identical to the allocating renderer; frame and scratch storage are reused after warmup; switching projection statistics preserves dimensions, spacing and labels; locked native/WASM checks, strict Clippy, rustdoc, formatting, real-study replay and bounded resource evidence pass.
- Dependency: RITK-SNAP-NATIVE-FRAME-REUSE-001 and the existing typed `SlabProjection`/native projection contract.
- Verification: native projection tests compare reusable output with the allocating oracle and pointer-swapped storage; the full native/WASM package gates, real 94-file MRI replay and three-run lifecycle resource record bind to the delivered source revision.
- Re-open trigger: a consumer requires interactive projection changes that alter the slab extent or a GPU projection path; those are separate typed seams.
- Delivery: RITK commit `9714a1ab70a14dcbbd8cc9b3fa1c5e1412c85556`; native scalar projection refreshes now reuse retained frame, scalar and RGBA storage for MIP, MinIP and Average.
- Verification: locked native nextest 482/482, strict native/WASM checks, feature Clippy, projection nextest 5/5, rustdoc/doctests, formatting, lockfile and script checks pass; the real 94-file MRI replay exits 0 with invalid-study exit 1; three bounded projection runs exit 0 with repeat capture SHA `056bf2cd...` and the resource record linked in the manual.


<a id="RITK-SNAP-OBLIQUE-RESLICE-001"></a>
## RITK-SNAP-OBLIQUE-RESLICE-001 — Resample physical viewer planes [arch] [minor]
- Status: done; priority: P1; owner: RITK rendering; integrator: root; last-update: 2026-09-20.
- Outcome: a validated physical-plane request produces an input-sensitive scalar plane with nearest-neighbour or trilinear samples and maximum, minimum or average slab reduction; the result remains format-neutral for Métis, eframe and VTK consumers.
- Scope: `ritk-snap` affine/reslice renderer, analytical/property tests, ADR and DICOM manual contract notes. DICOM parsing, host event wiring, GPU dispatch and clinical metadata remain in their existing owners.
- Acceptance: axis-aligned requests are byte/value-equivalent to existing slice extraction; rotated anisotropic requests preserve voxel↔patient coordinates; invalid basis, extent, spacing, sample count and out-of-volume requests return typed errors; slab statistics use the declared sample path; caller-owned scalar storage is reusable; locked native/WASM checks, strict Clippy, rustdoc, formatting and real-study replay pass.
- Dependency: existing `AffineTransform`, `LoadedVolume`, `SlabProjection`, VTK spatial-volume contract and Métis presentation-frame geometry.
- Verification: `ritk-snap` nextest 479/479, strict native Clippy, wasm32 check/Clippy, rustdoc, formatting, standalone lock and ADR-index checks pass; six new reslice tests cover axis equivalence, rotated anisotropic coordinates, trilinear linear-field recovery, slab statistics, capacity reuse, and typed invalid inputs.
- Delivery: RITK PR [#539](https://github.com/ryancinsight/ritk/pull/539), merge `2d9377975b4fc420ab0f16481c6e2a6c80279cc7`; physical-plane reslicing remains format-neutral for Métis, eframe and VTK consumers.

<a id="RITK-METIS-LOCK-014"></a>
## RITK-METIS-LOCK-014 — Replay the merged Métis semantic provider [patch]
- Status: done; priority: P1; owner: RITK viewer + integration; integrator: root; last-update: 2026-09-20.
- Outcome: the standalone RITK lock and browser replay resolve Métis at merge `b58d64b1bebe76bb32570339c4a349cc1b0d7086`, consume the host-neutral semantic provider, and preserve the real 94-file MRI framebuffer and invalid-study rejection.
- Scope: first-party lock resolution, the browser workflow's default revision, current real-study provenance, and focused native/WASM/replay verification. DICOM parsing and clinical semantics remain RITK-owned; historical evidence records stay bound to their generating revisions.
- Acceptance: standalone Cargo.lock resolves without the Atlas overlay; locked `ritk-snap` tests, strict native/WASM checks and Clippy, formatting, rustdoc, provenance and real-study replay pass; the 1280×800 PNG remains byte-identical (`259dd791...`, 411,589 non-black pixels); the browser workflow checks out the same full revision by default.
- Dependency: Métis PR #305 merge `b58d64b1bebe76bb32570339c4a349cc1b0d7086`; Moirai remains the current lock-pinned provider.
- Verification: lock SHA, source revisions, file count/bytes, image hash/dimensions/non-black count, invalid-study exit, native nextest, WASM check/Clippy, formatting, rustdoc, and browser workflow revision assertions are recorded in the provenance JSON and PR body.
- Delivery: RITK lock/workflow commit `940c0552eec0728c205ab57f361ec8f400b1e156`; current replay provenance records Cargo.lock SHA `d6f75fae8228e1a184ee2221a70f48478ae2388da7c3041eacb85cdc154a0afd`, six Metis sources at `b58d64b1bebe76bb32570339c4a349cc1b0d7086`, and the unchanged 1280×800 MRI PNG.

<a id="RITK-SNAP-NATIVE-FRAME-REUSE-001"></a>
## RITK-SNAP-NATIVE-FRAME-REUSE-001 — Reuse native orthogonal render storage [patch]
- Status: done; priority: P1; owner: RITK native presentation; integrator: root; last-update: 2026-09-20.
- Outcome: repeated native Métis orthogonal renders reuse caller-owned RGBA and transform storage after warmup while preserving the existing DICOM display pixels, geometry, and transforms.
- Scope: native orthogonal frame rendering, reusable transform output, value-semantic differential tests, native manual/resource evidence, ADR or design note if the public seam changes. DICOM decoding remains unchanged; scalar projection storage is delivered by RITK-SNAP-NATIVE-PROJECTION-REUSE-001, while browser/WebGPU projection remains a separate seam.
- Acceptance: the initial and subsequent native three-plane captures are byte-identical to the existing replay; scratch capacities remain stable after warmup; transformed dimensions and spacing remain correct for all view transforms; locked native/WASM checks, strict Clippy, rustdoc, formatting, real-study replay and resource evidence pass.
- Dependency: current `PresentationFrame`/`FrameRenderScratch` storage swap and the native Métis viewer workflow; no Metis or DICOM ownership change.
- Verification: locked native nextest 480/480 (including transformed frame capacity and RGBA differential tests), strict native/WASM Clippy, WASM check, rustdoc, formatting, standalone lockfile and Python checks pass; the real 94-file MRI replay exits 0 with invalid-study exit 1 and the byte-identical 1280×800 PNG (`259dd791...`, 411,589 non-black pixels); the JSON provenance records the lifecycle oracle and current binary/example hashes.
- Delivery: RITK commit `58f1fe8fdc69f99acc94fe0d0735a86d1a8bc9fa` is the source revision for the replay and manual evidence; native orthogonal frames now retain one presentation frame and transform scratch slot per plane, and the scalar projection frame and scratch now reuse storage under RITK-SNAP-NATIVE-PROJECTION-REUSE-001.

<a id="RITK-BROWSER-VIEWPORT-001"></a>
## RITK-BROWSER-VIEWPORT-001 — Present browser zoom and pan state [minor]
- Status: done; priority: P1; owner: RITK browser presentation; integrator: root; last-update: 2026-09-20.
- Outcome: browser Métis canvases display the RITK viewer's zoom and pan state while pointer actions map through the same transformed pixels.
- Scope: `ritk-snap` browser raster presentation, viewport coordinate mapping, focused value tests, ADR and DICOM manual; DICOM decoding, Metis APIs and native eframe composition remain out of scope.
- Acceptance: zoom/pan identity preserves RGBA bytes; zoom crops around the frame center; pan shifts pixels with black out-of-bounds; pointer mapping uses the same inverse transform; storage remains reusable after warmup; locked native/WASM tests, strict Clippy, formatting, rustdoc, real MRI replay and documentation checks pass.
- Dependency: existing `PresentationFrame` storage reuse, browser physical spacing and the host-neutral `SnapApp` zoom/pan policy.
- Correction: the post-merge review aligned the raster sampler with the existing image-edge coordinate contract; centered zoom now samples the same source pixels that the pointer inverse presents.
- Workflow correction: the hosted browser chooser now builds one populated argument vector, so empty projection and cine trace options are safe under Bash `nounset`.
- Verification: neutral locked `ritk-snap` nextest 473/473; strict native and wasm32 Clippy; wasm32 check; rustdoc; formatting; standalone lockfile check; Python script tests 22/22; real 94-file MRI replay passed with invalid-study exit 1 and byte-identical 1280×800 PNG (`259dd791...`, 411,589 non-black pixels).
- Delivery: RITK PR [#535](https://github.com/ryancinsight/ritk/pull/535), merge `695d33c7e675604b1fb32dfaffe3e90d9b0d21e0`; post-merge edge-coordinate correction PR [#536](https://github.com/ryancinsight/ritk/pull/536), merge `055b2886cb448bdc3d4a673ababf00d0def70b7d`; workflow argument correction PR [#537](https://github.com/ryancinsight/ritk/pull/537), merge `fa0ae3301ea2cc5e3733f7f1cb94fbaee2808f4c`. Hosted run [35512369724](https://github.com/ryancinsight/ritk/actions/runs/35512369724) passed the real-study bundle, four-cycle Chromium and Firefox rasters, Chromium window and MIP projection artifacts; Safari accepted 94 chooser paths but failed four bounded browser-read probes, and Chromium WebGPU had no adapter.

<a id="RITK-SNAP-BROWSER-PROJECTION-001"></a>
## RITK-SNAP-BROWSER-PROJECTION-001 — Present a selectable scalar projection in the browser [arch] [minor]
- Status: done; priority: P1; owner: RITK browser presentation; integrator: root; last-update: 2026-09-20.
- Outcome: the WASM Métis workflow can mount three interactive orthogonal canvases plus one display-only scalar projection canvas while RITK retains slab reduction, DICOM window/level, colormap and physical spacing.
- Scope: `ritk-snap` browser presentation, projection-statistic parsing, four-canvas startup exports, bounded projection semantics, frame-storage reuse, ADR, README and DICOM manual. Existing three-canvas and single-canvas exports remain unchanged; oblique resampling, browser WebGPU slab dispatch and DICOM parsing remain out of scope.
- Acceptance: maximum, minimum and average projection requests are validated before mount; the four-canvas raster workflow presents real scalar pixels with the selected label and dimensions; the explicit WebGPU workflow either presents through an adapter or surfaces its typed no-adapter setup error without raster fallback; the projection canvas has no input listeners; malformed requests and non-scalar studies surface typed errors; native/WASM checks, strict Clippy, rustdoc, formatting and browser-facing tests pass.
- Dependency: native scalar modes delivered by [RITK PR #526](https://github.com/ryancinsight/ritk/pull/526), merge `145372ea4781363abf28bad2790c3bfe25e1f202`; typed slab contract delivered by [RITK PR #525](https://github.com/ryancinsight/ritk/pull/525), merge `9658edd379ff561222d619d18e9d20a08860cbb4`.
- Verification: native and WASM projection surfaces passed the locked `ritk-snap` library suite (465/465), strict native/WASM Clippy, native/WASM checks, rustdoc, formatting and strict ADR-index validation; the four-canvas browser gallery now selects a real MIP, MinIP or Average surface through a bounded query parameter, verifies projection attributes and non-black RGBA pixels from the saved MRI study, and proves the display-only canvas does not increase the 3 × 7 provider listener budget.
- Delivery: RITK PR [#527](https://github.com/ryancinsight/ritk/pull/527), merge `9107da98cca72eebe2deffb466a6f5b7adb17948`; caller arguments were corrected by PR [#532](https://github.com/ryancinsight/ritk/pull/532), merge `a66d2ecee751c03bb111395ea239a10a88583a4c`; projection ordering was fixed by PR [#533](https://github.com/ryancinsight/ritk/pull/533), merge `f46d30091cac1ace12bb339d7421ac0c339ebf49`. Hosted run [35500085568](https://github.com/ryancinsight/ritk/actions/runs/35500085568) passed the Chromium MIP projection artifact [10601916572](https://github.com/ryancinsight/ritk/actions/runs/35500085568/artifacts/10601916572): the real 94-file study produced a 512 × 512, 110,028-pixel non-black projection with `consumer_listeners: 21` and `display_only: true`. WebGPU no-adapter and WebKit selected-file read remain explicit hosted limits.
- Re-open trigger: a consumer needs interactive projection gestures, oblique physical-plane resampling or GPU slab dispatch; each requires its own typed contract.

<a id="RITK-VTK-SPATIAL-VOLUME-001"></a>
## RITK-VTK-SPATIAL-VOLUME-001 — Preserve physical volume geometry across the VTK boundary [arch] [minor]
- Status: done; priority: P1; owner: RITK volume/VTK integration; integrator: root; last-update: 2026-09-20.
- Outcome: a loaded clinical volume crosses into a zero-copy, VTK-compatible spatial volume contract with dimensions, origin, spacing, direction, channels and scalar payload preserved; existing VTK serialization remains an explicit materialization boundary.
- Scope: `ritk-vtk` spatial-volume contract, `ritk-snap` conversion from `LoadedVolume`, value-semantic tests, ADR and VTK integration documentation. Oblique resampling, slab projection and Metis rendering are follow-up increments; DICOM parsing remains in `ritk-snap`/`ritk-io`.
- Acceptance: anisotropic rotated geometry and channel-fastest scalar order survive conversion; malformed shape, spacing, direction and payload inputs return typed errors; the conversion shares the source allocation; an explicit materialization produces a valid `VtkImageData`; locked package tests, strict Clippy, formatting, rustdoc and diff checks pass.
- Dependency: existing direction-aware `LoadedVolume` and `ritk-vtk` `VtkImageData`; no dependency on Metis or GUI semantics.
- Verification: full `cargo nextest run -p ritk-vtk -p ritk-snap --lib` passed 719/719 under the Atlas overlay; focused VTK (4/4) and conversion (2/2) value tests passed; strict package Clippy, rustdoc, formatting, standalone lock and diff checks pass.
- Evidence: anisotropic rotated geometry, channel-fastest payload, zero-copy pointer identity, malformed geometry/payload partitions and explicit `VtkImageData` materialization are covered in the new tests; standalone lock resolves 61 first-party git sources at current Metis `38b2de4` and Moirai `2a54e01` revisions.
- Delivery: RITK PR [#523](https://github.com/ryancinsight/ritk/pull/523), merge `07f23d4831b9c0b493102b3aa3f05d3da4d23c68`.

<a id="RITK-BROWSER-CINE-REPLAY-001"></a>
## RITK-BROWSER-CINE-REPLAY-001 — Reproduce the current browser cine residual [patch]
- Status: done; priority: P1; owner: RITK browser presentation; integrator: root; last-update: 2026-09-20.
- Outcome: current lock-pinned Chromium consumer replay presents changed real-study slices through Play and rate changes, preserves live timeout diagnostics, and completes one shared cine/tool teardown; Safari bounded file reads and WebGPU adapter availability remain separate residuals.
- Scope: `ritk-snap` browser viewer/presentation and bounded gallery cine diagnostic; DICOM decoding, Metis file access, and WebGPU provider selection remain out of scope.
- Acceptance: the hosted current-pair replay reaches one presented slice transition after Play and after the rate change, with generation/index semantics and real MRI pixels; timeout failures retain live canvas/control state and viewer status; locked native/WASM tests, strict Clippy, formatting, docs and browser script tests pass.
- Basis: hosted run [35487777698](https://github.com/ryancinsight/ritk/actions/runs/35487777698), RITK `4cbef46ef4be41fe2a93bcb79c81a02d2c73d685`, Metis `165c4ec923e76ea7bc32b6b4fb99b4338166b3a3`; the Chromium-window job passed after the 94-file replay, while WebKit and WebGPU retained independent residuals.
- Delivery: [PR #522](https://github.com/ryancinsight/ritk/pull/522), current branch head `4cbef46ef`; the bounded cine capture now validates generation/index snapshots and shares final teardown with the tool capture.
- Evidence: Chromium-window [artifact 10597873405](https://github.com/ryancinsight/ritk/actions/runs/35487777698/artifacts/10597873405) records `trace.json` status `passed`, 94 files/49,807,236 bytes, sagittal 255→256 on Play and 256→257 after the 24 FPS rate change, and final `mounted=false`/zero listeners with `teardown.owner=tool-controls`; the 2,880×2,114 `gallery-cine.png` has SHA-256 `73542c9c7be8ff1da18e90faaefa5105ff1e1edb94f2a1eef067d313daca0e03` and visibly contains the three non-black MRI planes.
Unresolved delivery items are kept as executable records. Closed history is indexed below; full prose remains in git.

<a id="RITK-SNAP-PRESENTATION-FRAME-REUSE-001"></a>
## RITK-SNAP-PRESENTATION-FRAME-REUSE-001 — Reuse host presentation frame storage [patch]
- Status: done; priority: P1; owner: RITK presentation; integrator: root; last-update: 2026-09-20.
- Outcome: repeated browser slice presentation reuses bounded RGBA storage after warmup while preserving byte-identical pixels and physical display spacing; the reusable frame API is available to native callers.
- Scope: `ritk-snap` presentation frame and slice-render scratch seam, browser surface cache, tests and DICOM manual evidence; DICOM decoding, clinical semantics, native-session composition and Metis ownership remain unchanged.
- Acceptance: repeated browser-study slice updates retain frame-slot capacity without per-frame RGBA allocation; native/WASM checks, strict Clippy, locked presentation tests, rustdoc, formatting and real MRI pixel/hash oracles pass; no ranking claim is made.
- Delivery: browser surfaces retain `PresentationFrame` slots and swap bounded RGBA storage with reusable extraction scratch; study replacement returns frame storage to scratch; native session composition remains unchanged.
- Verification: locked native nextest 455/455 and eframe-shell nextest 867/867; strict native and wasm32 Clippy; wasm32 checks with and without eframe-shell; rustdoc; formatting; standalone lockfile and diff checks; real 94-file MRI replay exit 0 with invalid-study exit 1 and byte-identical 1280×800 PNG (`259dd791...`, 411,589 non-black pixels). Hosted Linux Clippy exposed unused Windows-only compositor helpers after merge; the fix-forward scopes both helpers to Windows before the next hosted gate.
- Evidence: current executable SHA `b260611...` (54,011,904 bytes); the reusable-frame regression asserts byte identity and stable frame/scratch capacities after warmup. Hosted browser residuals remain tracked by the existing cross-engine evidence item; this increment makes no framework or engine ranking claim.

<a id="RITK-SNAP-PRESENTATION-GEOMETRY-001"></a>
## RITK-SNAP-PRESENTATION-GEOMETRY-001 — Bind physical geometry to host-neutral frames [arch] [minor]
- Status: done; priority: P1; owner: RITK presentation; integrator: root; last-update: 2026-09-20.
- Outcome: `PresentationFrame` carries one validated `PresentationSpacing` value, and native/browser hosts consume it for aspect validation and semantics without duplicating voxel-spacing derivation.
- Scope: `ritk-snap` presentation frame, native session layout, browser geometry/semantics, ADR and manual/API documentation; DICOM decoding and VTK representation remain out of scope.
- Acceptance: axis-specific anisotropic spacing survives slice construction and transformed native frames; browser aspect and native placement consume the same frame metadata; malformed geometry is rejected; locked native/WASM tests, strict Clippy, formatting, rustdoc and real MRI replay pass.
- Dependencies: Metis `METIS-PRESENTATION-GEOMETRY-001` provider merge; existing RITK physical-aspect and real MRI capture oracles.
- Delivery: [PR #518](https://github.com/ryancinsight/ritk/pull/518), merge `694904718d7ec922883ab2a6b572e9aa29edab99`; source and lock integration commit `b432ae69`; replay/provenance follow-up is included in the merged delivery.
- Verification: native `ritk-snap` nextest 454/454; strict native Clippy; wasm32 release check and strict Clippy; rustdoc with `-D warnings`; lockfile and Python script tests 12/12; actual 94-file MRI replay exit 0 with invalid-study exit 1 and byte-identical 1280×800 PNG (`259dd791...`, 411,589 non-black pixels).
- Evidence: standalone lock SHA `4f4b96958a0545203775e21b3bd1152a864bd6b77847d2b638213d6b420b1084`; native executable `b260611...` (54,011,904 bytes); example `78e2d1...` (24,153,600 bytes); source/provider revisions are recorded in `docs/manual/images/dicom-metis-real-mri.json`.

<a id="RITK-SNAP-EFRAME-CURRENT-001"></a>
## RITK-SNAP-EFRAME-CURRENT-001 — Refresh the matched eframe MRI baseline [patch]
- Status: done; priority: P1; owner: RITK viewer + integration; integrator: root; delivery: [PR #517](https://github.com/ryancinsight/ritk/pull/517), merge `19c3ec42f0f7f48a0f6dfb91337e828ecfadf965`; last-update: 2026-09-19.
- Outcome: re-run the existing shell-free eframe orthogonal presentation against the current standalone RITK lock and bind the revision, executable, capture and resource metrics to the real 94-file MRI fixture.
- Acceptance: three bounded runs exit 0, repeat capture SHA matches, semantic surfaces and physical host extent remain axial/coronal/sagittal at 1280×800, real-image capture is visually inspected, and no framework ranking is claimed.
- Delivery: current locked eframe build; three bounded lifecycle runs exited 0 with repeated source SHA `2e47199cca0851f5ca2e3ce7b613cf70b096c3528bfc1eb705f64ced75485ba9`; mean peak private bytes `416,867,669 ± 30,202,762`, mean lifecycle duration `2,123 ± 203 ms`; provenance binds RITK lock commit `f1a556786e849696caa73d8c341adf309e87d163`, Metis `8d4ab58e8731c51547bbca3ec87100facb698322`, and Moirai `f038622d24907884ce5f386da4e04d05bdb60d62`.
- Verification: `scripts/resource.py` lifecycle report; real-image visual inspection; JSON/hash/diff checks; standalone lock check; current eframe build.
- Evidence: the Metis resource runner recorded three orderly exits, matching capture SHA `2e47199cca0851f5ca2e3ce7b613cf70b096c3528bfc1eb705f64ced75485ba9`, 1280×800 source pixels with 927,849 non-black pixels, mean peak private bytes `416,867,669 ± 30,202,762`, mean lifecycle duration `2,123 ± 203 ms`, and peak handles `515`. The capture visibly contains the actual axial, coronal and sagittal MRI planes; no framework ranking is inferred.

<a id="RITK-METIS-LOCK-012"></a>
## RITK-METIS-LOCK-012 — Replay the merged Metis clipboard runtime [patch]
- Status: done; delivery: RITK PR [#510](https://github.com/ryancinsight/ritk/pull/510), merge `e4eb95b5769e51066533c52ac32fc70306a9224e`; compacted 2026-09-18.
- Outcome: standalone lock binds Metis `d7cb62f`, Moirai `b179b89`, and lock `2194947d`; neutral locked nextest 450/450, native/WASM checks, strict Clippy, formatting, rustdoc and replay gates pass. The public 94-file MRI study reads 49,807,236 bytes, rejects the invalid study with exit 1, and reproduces the 1280×800 image byte-identically (`259dd791...`, 411,589 non-black pixels); DICOM ownership remains in RITK.

<a id="RITK-METIS-LOCK-013"></a>
## RITK-METIS-LOCK-013 — Replay the merged Metis display-scale probe [patch]
- Status: done; priority: P1; owner: RITK viewer + integration; integrator: root; delivery: [PR #515](https://github.com/ryancinsight/ritk/pull/515), merge `9881b9c4ac9fdccbc241c94fe1903f30d4f20e95`; last-update: 2026-09-19.
- Outcome: standalone Cargo.lock and the real 94-file MRI replay advance all Metis packages to merge `8d4ab58e8731c51547bbca3ec87100facb698322` and Moirai packages to merge `f038622d24907884ce5f386da4e04d05bdb60d62`, preserving the byte-identical clinical framebuffer while consuming the delivered physical-monitor probe and per-monitor DPI provider.
- Acceptance: standalone lock resolves without the Atlas overlay; locked native/WASM tests, strict Clippy, formatting, rustdoc, provenance and the real MRI replay pass; DICOM parsing and presentation remain RITK-owned.
- Dependency: Metis PR #286 merged with hosted Windows gate `35471442071`; Moirai PR #406 merged at `f038622d24907884ce5f386da4e04d05bdb60d62` with local workspace gates green.
- Evidence: lock SHA-256 `2825653169ee324f9421122416da1e65a279e8657841c7fce9347386f6b58745`; the existing real-study harness exits 0 after reading 94 files/49,807,236 bytes, its invalid-study probe exits 1, and the actual three-plane PNG remains byte-identical (`259dd79103482756c4e688621bebafc841cc40f1df10ff2bbd7f9d04b7b4d401`, 1280×800, 411,589 non-black pixels). Locked native nextest is 450/450; strict native/WASM checks and Clippy, formatting, lock validation and rustdoc pass.

<a id="RITK-METIS-LOCK-011"></a>
## RITK-METIS-LOCK-011 — Replay the current Metis consumer lock [patch]
- Status: done; delivery: RITK PR [#508](https://github.com/ryancinsight/ritk/pull/508), merge `632f22219`; compacted 2026-09-19.
- Outcome: standalone lock and real-study provenance bind Metis `b432446`, Moirai `5075d4c`, lock `182203ea`, and the byte-identical 1280×800 MRI frame; locked native/WASM, strict Clippy, formatting, rustdoc and 450/450 nextest gates pass, with invalid-study exit 1.

<a id="RITK-SNAP-METIS-NATIVE-WL-001"></a>
## RITK-SNAP-METIS-NATIVE-WL-001 — Demonstrate native window/level interaction [minor]
- Status: done; delivery: RITK PR [#503](https://github.com/ryancinsight/ritk/pull/503), merge `bf2526e90055bb8e6efbaa36d0f084baab2689da`; compacted 2026-09-18.
- Outcome: the native Métis session regression drives the existing RITK W/L pointer reducer against a loaded DICOM fixture, proves changed center/width and presented pixels, and verifies idle teardown; the user manual documents the gesture and keeps DICOM ownership in RITK. Local focused nextest, format, lockfile, provenance, and pre-push gates passed; hosted CI runs [35400891419](https://github.com/ryancinsight/ritk/actions/runs/35400891419) and [35400890913](https://github.com/ryancinsight/ritk/actions/runs/35400890913) passed.

<a id="RITK-METIS-LOCK-010"></a>
## RITK-METIS-LOCK-010 — Replay the merged Metis keyboard runtime [patch]
- Status: done; delivery: RITK PR [#506](https://github.com/ryancinsight/ritk/pull/506), merge `fafe9c90a6d91cb646ec5e0472f537de4e3b4b03`; compacted 2026-09-19.
- Outcome: the standalone lock, browser workflow and manual now bind Metis `860ffbf52d12a70c80dd2f150d2aff3a5630d57e` and Moirai `5075d4c70ba4f840d4c5a47b67c5d564405badf5`; the real 94-file MRI replay reads 49,807,236 bytes, rejects the invalid study, and reproduces the 1280×800 image byte-for-byte. Locked nextest 488/488, native/WASM checks and strict Clippy, formatting, docs, provenance and image gates pass; DICOM ownership remains in RITK.

<a id="RITK-METIS-LOCK-009"></a>
## RITK-METIS-LOCK-009 — Advance the current Métis and Moirai provider lock [patch]
- Status: done; delivery: RITK PR [#504](https://github.com/ryancinsight/ritk/pull/504), merge `d8ed71cc6b592fe7a80cb4710d616aa1ffab5c1`; compacted 2026-09-18.
- Outcome: the standalone lock and current browser workflow/manual pins now resolve Metis `82bb3af7bb1695b43767caf9cf1012273e58613d` and Moirai `5075d4c70ba4f840d4c5a47b67c5d564405badf5`; locked native/WASM gates and the fresh 94-file MRI replay pass with the committed image byte-identical. Historical provenance remains bound to its generating revisions; DICOM ownership stays in RITK.

<a id="RITK-METIS-LOCK-008"></a>
## RITK-METIS-LOCK-008 — Advance the WebGPU recovery provider pins [patch]
- Status: done; delivery: RITK PR [#499](https://github.com/ryancinsight/ritk/pull/499) (`d7ebd97dd`), [#500](https://github.com/ryancinsight/ritk/pull/500) (`c34f05d46`), [#501](https://github.com/ryancinsight/ritk/pull/501) (`9fa6c12d4`); compacted 2026-09-18.
- Outcome: the standalone 61-source lock, native/WASM gates, and real 94-file MRI replay pass; hosted run [35395627386](https://github.com/ryancinsight/ritk/actions/runs/35395627386) records passing Chromium/Firefox captures plus the WebKit bounded-read, WebGPU no-adapter, and application-window cine residuals. Details: [DICOM manual](docs/manual/dicom-workflow.md) and [cross-engine provenance](docs/manual/images/dicom-metis-real-browser-mri-cross-engine.json).

<a id="RITK-METIS-LOCK-007"></a>
## RITK-METIS-LOCK-007 — Advance the current Metis text provider pin [patch]
- Status: done; priority: P1; owner: RITK viewer + integration; integrator: root; last-update: 2026-09-18.
- Scope: advance the six Metis package sources from `8f33126f23b8dc327bea45c4fe513a6b60b73c99` to merged Metis `0e449856ade677860fd6866c3855ae2ba527e33a`, then replay the real MRI surface and rebind the standalone provenance.
- Acceptance: standalone `Cargo.lock` resolves with 61 first-party sources; locked `ritk-snap` tests, native/WASM checks and strict Clippy pass; the 1280×800 real MRI PNG remains byte-identical; docs name the exact lock digest and provider revisions.
- Dependency: Metis PR #258 merged with hosted Windows/artifact gates; DICOM parsing and clinical presentation remain RITK-owned.
- Delivery: lock commit `e635baf90f99cc0ae09df06c42424ba5c4c6faec`; six Metis sources resolve at `0e449856ade677860fd6866c3855ae2ba527e33a`; standalone lock SHA-256 `49381c64b5751b5c07bf571c66a31205ebf3ccdc780afe3b4b102c0792a5bc85`.
- Evidence: rebuilt `ritk-snap` exits 0 on the 94-file public MRI-DIR study and the invalid-study probe exits 1; PNG SHA `259dd79103482756c4e688621bebafc841cc40f1df10ff2bbd7f9d04b7b4d401`, 1280×800, 411,589 non-black pixels; executable SHA `210a6a0e0f6efa59d76ad91de8a6d068d12b9e7796df17c00d3be270e829208c` (25,973,760 bytes). Locked native nextest exits 0 for 449 tests; strict native Clippy, WASM check/Clippy, format, lockfile and provenance JSON checks pass.

<a id="RITK-BROWSER-READ-001"></a>
## RITK-BROWSER-READ-001 — Reproduce and close WebKit study reads
- Status: blocked; compacted 2026-09-18; full delivery history remains in git.
- Scope: saved public MRI-DIR chooser replay and cross-engine gallery; provider/host fixes stay with their owners.
- Acceptance: read-path cause and production fix, 94 file hashes, three exact pixel oracles, bounded rejections and clean sessions on Chromium, Firefox and WebKit.
- Risk: [patch]; dependency: [Metis read diagnosis](../metis/backlog.md#METIS-BROWSER-READ-001).
- Delivery: RITK PR [#422](https://github.com/ryancinsight/ritk/pull/422), merge `18ee4e55b`; the current failure evidence is recorded and the external WebKit authorization residual remains blocked.
- Evidence: current hosted [run 35535831906](https://github.com/ryancinsight/ritk/actions/runs/35535831906) builds RITK `651b4a808374afc3efca35b71b9583b0a2a17248` against Metis `b58d64b1bebe76bb32570339c4a349cc1b0d7086`; Chromium and Firefox pass the 94-file real MRI study, three exact pixel oracles and lifecycle checks; WebKit accepts the chooser but denies the first bounded read under Safari 26.6.2, with diagnostics preserved in the provenance JSON.
- Diagnosis: the current WebKit trace shows the sandbox rejecting the bounded whole-file read after chooser acceptance; earlier isolated one-file/full-batch probes also failed across the available browser read paths. DICOM stays in RITK.
- Verification: locked `ritk-snap` nextest 487/487, strict native/WASM Clippy and checks, formatting, rustdoc and lockfile validation pass; [proof and log hashes](docs/manual/images/dicom-metis-real-browser-mri-cross-engine.json).
- Blocker: exact SafariDriver/WebKit selected-file authorization defect remains external; recorded run 35535831906 reproduces the denial after SafariDriver accepts all 94 files (artifact [10612972422](https://github.com/ryancinsight/ritk/actions/runs/35535831906/artifacts/10612972422), diagnostics [10613371829](https://github.com/ryancinsight/ritk/actions/runs/35535831906/artifacts/10613371829)). Re-open when the corrected browser/runner path grants real-file reads; application byte-read APIs cannot grant that access.

<a id="RITK-DOCS-EVIDENCE-SYNC-002"></a>
## RITK-DOCS-EVIDENCE-SYNC-002 — Rebind current real MRI replay provenance [patch]
- Status: done; delivery: RITK PR #493, merge `73b01be1a3fcb6bb755ffdd302cb412d218fcd44`; compacted 2026-09-18.
- Outcome: standalone-lock provenance binds RITK `e635baf90f99cc0ae09df06c42424ba5c4c6faec`, Metis `0e449856ade677860fd6866c3855ae2ba527e33a`, Moirai `a2f21496d1d09b2abe6523e3c8cdbf751dcd560a` and lock `49381c64b5751b5c07bf571c66a31205ebf3ccdc780afe3b4b102c0792a5bc85`; the 94-file MRI replay remains byte-identical at `259dd79103482756c4e688621bebafc841cc40f1df10ff2bbd7f9d04b7b4d401` (1280×800, 411,589 non-black pixels), with JSON/image/hash and locked gates validated.

<a id="RITK-BROWSER-WEBGPU-001"></a>
## RITK-BROWSER-WEBGPU-001 — Demonstrate browser WebGPU presentation [arch] [minor]
- Status: blocked; compacted 2026-09-18; full delivery history remains in git.
- Scope: replay the saved public MRI-DIR study through the RITK-owned `?renderer=webgpu` page and retain actual canvas/window evidence; RITK owns DICOM decoding and clinical pixels, while Métis remains the format-neutral canvas host.
- Acceptance: a configured browser runner reports an adapter, presents the three saved-study canvases, records revision-bound PNGs and semantic attributes, and completes bounded teardown without a raster fallback.
- Blocker: hosted Chromium in run [35535831906](https://github.com/ryancinsight/ritk/actions/runs/35535831906) reports no WebGPU adapter; the current setup error and failure capture artifact [10613591260](https://github.com/ryancinsight/ritk/actions/runs/35535831906/artifacts/10613591260) are preserved in [`dicom-metis-real-browser-mri-webgpu-failure.png`](docs/manual/images/dicom-metis-real-browser-mri-webgpu-failure.png).

<a id="MIG-439-03"></a>

## MIG-439-03 — Replace remaining Burn NdArray backend aliases with Atlas-backed surfaces

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Atlas-backed surfaces. RESCOPED (was READY; original acceptance criteria do not hold — see Sprint 465 finding).** **Correction (Sprint 465, evidence-based)**: investigated the strongest candidate crate (`ritk-jpeg`, smallest burn_ndarray footprint, already has a parallel Coeus reader) to execute th…

<a id="PERF-435-01"></a>

## PERF-435-01 — Route MSE through fused interpolation. PARTIAL.

- Status: in-progress; retained from the historical execution section; full detail remains in git.

- Scope: Generalized `ritk_interpolation::transform_and_interpolate` over spatial dimensionality, generalized the OOB mask helper over the image shape length, and routed `MeanSquaredError` through the fused transform-to-index-to-linear interpolation path. Evidence tier: value-semantic nextest and focused ti…

<a id="PROVIDER-420-01"></a>

## PROVIDER-420-01 — Hermes complex dispatch bound cleanup. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: The local Atlas provider graph exposed that Hermes complex SIMD operations still require `Neg` at the complex-operation dispatch surface after broader unsigned scalar support. A minimal local fix passes `cargo check -p hermes-simd --all-targets`; full provider rustfmt is still blocked by unrelated…

<a id="PERF-419-01"></a>

## PERF-419-01 — Registration test runtime budget breach. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 419's `ritk-registration` nextest gate passed but exposed integration tests above the 30s slow budget, including 100s, 146s, and 193s rows. Treat this as a real performance defect to profile; do not weaken or skip those tests.

<a id="COEUS-406-01"></a>

## COEUS-406-01 — Fix dirty Coeus autograd provider compile break. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: RITK doctest/doc gates against the current local Atlas stack are blocked after refreshing Coeus path packages to `0.2.6`: `D:\atlas\repos\coeus` is dirty on `test/cuda-parity-suite`, and `coeus-autograd` fails to compile in shape/reduction ops. This must be fixed in Coeus before RITK can claim docs…

<a id="PERF-406-02"></a>

## PERF-406-02 — Registration test runtime budget breach. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 406's touched-package `nextest` gate passed but exposed registration tests above the 30s slow budget, including 93s, 129s, and 183s rows. Treat this as a real performance defect to profile; do not weaken or skip those tests.

<a id="PERF-387-02"></a>

## PERF-387-02 — Continue flat-buffer memory-efficiency audit. IN PROGRESS.

- Status: in-progress; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 387 flattened `VectorConfidenceConnected` covariance/inverse matrices and removed the B-spline legacy placeholder. Sprint 389 flattened `InverseDisplacementField` TPS spline/affine coefficient blocks after the solve. Sprint 390 flattened TIFF grayscale/RGB page accumulation by removing `Vec<…

<a id="MIG-387-01"></a>

## MIG-387-01 — Atlas crate migration audit.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Continue replacing production `nalgebra`/`ndarray`/`burn` surfaces with `leto`/`coeus`/ `hephaestus` only after each target operation has a verified equivalent contract and focused differential tests. Do not remove boundary dependencies used only for file-format interop or external framework contra…

<a id="MIG-387-02"></a>

## MIG-387-02 — Spatial Leto SSOT migration. IN PROGRESS.

- Status: in-progress; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 408 migrates `ritk-spatial` storage to Leto fixed vectors/matrices and removes direct `nalgebra` dependencies from `ritk-core`, `ritk-metaimage`, `ritk-nrrd`, `ritk-nifti`, and `ritk-mgh` spatial direction setup. Sprint 409 moves DICOM IO, MINC, and filter spatial-transform consumers onto `D…

## Sprint 342 (Phase 20) — Coeus Migration Readiness Audit

- Status: in-progress; historical sprint retained only for unresolved rows.

- **Status**: In Progress

- | MIG-342-04 | RITK-owned tensor contract over Coeus CPU backend | **Open** |

- | GPU-342-05 | Coeus WGPU differential test harness for RITK operation subset | **Open** |

- | REG-342-06 | Registration autodiff tape continuity proof/test under Coeus | **Open** |

- | MODEL-342-07 | `ritk-model` Coeus module/parameter/3-D convolution migration design | **Open** |

- | PY-342-08 | Python binding conversion plan over Coeus-backed Rust core | **Open** |

- The next implementation stage is not a dependency swap. It is the RITK tensor

- ### Residual risks



## Sprint 332 (0.50.95) — Documentation Compaction + Structural Audit + Benchmark

- Status: in-progress; historical sprint retained only for unresolved rows.

- **Status**: In Progress

- | BENCH-332-03 | `STACK_WEIGHTS_CAPACITY=32` Criterion benchmark — measure AVX2 speedup vs 8-entry version | **Open** |

- | GPU-332-04 | Evaluate `sparse.rs` GPU-backend potential (Burn autodiff scatter compatibility, custom kernel feasibility) | **Open** |

- | CRLF-332-05 | Git CRLF normalization (`git add --renormalize`) — blocked by missing test data files | **Blocked** |



<a id="RITK-GAP-2026-08-20-01"></a>
## RITK-GAP-2026-08-20-01 [major][arch] — collapse the dual `X` / `X_native` surface
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-02"></a>
## RITK-GAP-2026-08-20-02 [minor] — fuzz the sixteen format parsers
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-03"></a>
## RITK-GAP-2026-08-20-03 [patch] — retire the GPU naming and the unbacked accelerator claims
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-04"></a>
## RITK-GAP-2026-08-20-04 [patch] — evict the 1.6 GB tracked binary payload
- Status: in-progress; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-05"></a>
## RITK-GAP-2026-08-20-05 [patch] — derive the escalated test budgets and sweep the dead filters
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-06"></a>
## RITK-GAP-2026-08-20-06 [patch] — raise the lint and documentation floor
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-07"></a>
## RITK-GAP-2026-08-20-07 [patch] — restore the CHANGELOG version axis
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-08"></a>
## RITK-GAP-2026-08-20-08 [patch] — write the registration and dispatch book chapters
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-09"></a>
## RITK-GAP-2026-08-20-09 [patch] — derive or remove the MI subsample stride
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.
