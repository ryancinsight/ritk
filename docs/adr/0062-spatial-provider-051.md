# ADR 0062: RITK spatial provider 0.51

## Status

Accepted for the RITK 0.5.0 spatial release on 2026-11-19.

## Context

The published Spatial 0.4.0 uses Leto 0.46 types in the conversion APIs.
A fresh registry build of that archive can select an Aequitas and Eunomia
combination that cannot build Leto. Consumers also need one nominal array
type boundary while moving to Coeus 0.13.0 and Eunomia 0.12.1.

## Decision

Version Spatial at 0.5.0 and require Leto 0.51.0. Advance all internal
Spatial requirements with it. A second Leto type family would force
duplicate conversions throughout the imaging and geometry APIs.

The geometry and format contracts remain in force. Consumers use the
same physical coordinates and on-disk DICOM conventions through the
new Leto provider version.

Diffusion Scheme's public direction and weighting APIs name Spatial and
Aequitas types. Version it at 0.4.0 with Spatial 0.5.0 and Aequitas 0.4.0.
DICOM must resolve the new Diffusion Scheme release so both paths share
the same Spatial identity.

## Verification

Run the Spatial geometry and DICOM parsing suites. Build and compare the
published API baseline using its archived lock. Verify the new Spatial
archive against crates.io and use it from a standalone DICOM build.
