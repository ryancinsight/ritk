# ADR 0034: Native Métis projection layout

Status: Accepted

Date: 2026-09-12

Driver: [RITK-SNAP-METIS-MIP-001](../../backlog.md#RITK-SNAP-METIS-MIP-001).

## Context

The RITK eframe viewer presents a loaded scalar study as three orthogonal
planes plus a CPU or GPU axial maximum-intensity projection (MIP). The native
Métis session currently presents only the three orthogonal planes. Comparing
those captures therefore measures different workloads and cannot support a
semantic or resource comparison.

DICOM opening, grayscale presentation, projection computation, and viewer state
belong to RITK. Métis owns the native surface and accepts only a bounded RGBA
frame. The existing orthogonal layout and its pointer routing are already
validated and must remain the default.

## Decision

Add a typed `NativePresentationMode` with `Orthogonal` as the default and
`OrthogonalWithMip` as an explicit native launch option. RITK renders the
existing scalar axial MIP through the same window/level and colormap policy as
eframe, converts it to the format-neutral `PresentationFrame`, and composes it
in the fourth panel of a bounded 2×2 native framebuffer. The three orthogonal
panels retain their existing viewports and event routing; the MIP panel is a
display-only projection.

Color volumes continue to follow the existing RITK scalar-projection error
contract. The option is native-only and does not add DICOM or patient data to
Métis. The default `Orthogonal` path keeps its existing dimensions and pixel
layout so prior captures remain valid.

## Rejected alternative

Adding a DICOM or projection implementation to Metis would move clinical
semantics across the host boundary and duplicate RITK behavior. Reusing the
eframe `LayoutMode` would couple the native host to the legacy UI state and
would not provide a format-neutral frame. A second MIP algorithm in the native
session would make eframe and Métis diverge; the existing RITK renderer is the
single projection implementation.

## Verification

Unit tests cover the typed mode, default orthogonal byte behavior, four-panel
placement, projection pixel equivalence with the existing RITK MIP renderer,
and color-volume rejection. Locked `ritk-snap` tests, strict Clippy, Rustdoc,
doctests, and the Windows executable workflow run against the delivered
revision. A real public MRI-DIR CT capture records the four panels and its
SHA-256 in the user manual; private DICOM captures remain local.
