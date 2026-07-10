# Reproducible Neural Debris v126

The approved scope combines a portfolio README, experimental notebook archive, and a Python reproduction of heon29/neural-debris version 126 (NB51). Preserve notebook computation order, formulas, defaults and gate-controlled CSV exports. Externalize input/output paths and configuration, remove automatic package installation from the computation pipeline, and retain the downloaded source with provenance.

First extract executable Python stages, preserving the notebook's shared namespace explicitly rather than changing mathematical dependencies during conversion. The CLI validates configuration and data before executing stages, and offers inspection without GPU imports. CPU tests verify extraction fidelity, stage execution semantics and configuration validation. Full training requires a compatible CUDA/Detectron2 environment and is reported separately from CPU verification.

Experiment notebooks are collected in `notebook/experiments/` at the author's request; test paths and archive navigation follow this location. Original notebook contents remain preserved. GitHub publication follows repository review and destination selection.
