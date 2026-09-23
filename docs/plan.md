# PCDSegmentation Rebuild Plan

**Goal:** Incrementally rebuild semantic point-cloud segmentation, training, evaluation, and preprocessing around the existing RandLANet implementation while preserving `SegmentClass` and all documented script workflows.

**Root-facing contract:** `SegmentClass(...).segment_pcd(points, intensity)` returns one point-aligned label array.

**Design:** `../../../docs/rebuild.md`

**Branch requirement:** Perform all rebuild work in this repository and its nested `nn_utils` submodule on `development`. Verify both branches first and request explicit approval before creating or switching either one.

## Task 1: Establish the uv project

**Files:** create `.python-version`, `pyproject.toml`, `uv.lock`; update `.gitignore` and README installation commands.

- [x] Replace the legacy requirements files as sources of truth with minimal direct dependencies.
- [x] Define `basic` for inference/headless preprocessing and `test` including `basic`, `pytest`, `matplotlib`, and `pyvista`.
- [x] Keep plotting imports out of normal inference unless a plotting operation is explicitly requested.
- [x] Configure PyTorch 2.14/Torchvision 0.29 CPU and CUDA 13.2 profiles without pinning NVIDIA transitive wheels.
- [x] Preserve and verify the nested `nn_utils` submodule/package relationship.
- [x] Verify clean basic/test syncs and current public imports.

## Task 2: Protect interfaces and invocations

**Files:** add unit tests beside owned functions; create `tests/test_invocation.py` and integration tests under `tests/`.

- [ ] Characterize `SegmentClass` construction, model/config lookup, input validation, chunking/overlap, scaling, device use, and output ordering.
- [ ] Test empty and malformed point/intensity arrays, missing artifacts, provider errors, and deterministic small inference doubles.
- [ ] Pin CLI flags/defaults and file discovery/output behavior in `src/main.py` and preprocessing scripts.
- [ ] Test imports from the submodule root and the parent BRIK root.
- [ ] Test direct and module forms for preprocessing, inference, training, and evaluation help/test modes.

## Task 3: Normalize package imports

**Files:** current modules under `src/`; new internal package modules only where a tested responsibility is extracted.

- [ ] Replace `sys.path` mutation and generic `from utils` imports with package-relative imports.
- [ ] Keep existing script files as compatibility wrappers so documented direct execution still works.
- [ ] Replace wildcard imports with explicit consumed names.
- [ ] Resolve configs, weights, datasets, and result paths from explicit arguments or stable module-relative roots.
- [ ] Verify every invocation form after each import change.

## Task 4: Separate inference responsibilities

- [ ] Extract model artifact/config loading behind the existing `SegmentClass` constructor contract.
- [ ] Separate voxel partitioning, scaling, model prediction, and result merging one tested step at a time.
- [ ] Keep array shapes, overlap semantics, stable ordering, and labels unchanged.
- [ ] Make CPU/CUDA device placement explicit and prevent mixed-device tensors.
- [ ] Retain `src/array_processing.py` as the root-compatible import layer.

## Task 5: Separate offline workflows

- [ ] Isolate LAZ preprocessing and HDF5 dataset creation from inference imports.
- [ ] Isolate training configuration, Optuna orchestration, one-run training, and reporting.
- [ ] Isolate evaluation loading, metrics, and output plotting.
- [ ] Preserve current JSON schemas, CLI arguments, output naming, and result directories.
- [ ] Ensure plotting dependencies are loaded only by plotting/reporting paths.

## Task 6: Verify

- [ ] Run focused unit tests after each extraction and the complete suite at each task gate.
- [ ] Run direct/module invocation tests from documented working directories.
- [ ] Run a deterministic CPU inference smoke test using a small fixture.
- [ ] Run a production-weight CUDA smoke test on supported Linux hardware and record the effective device.
- [ ] Run the root BRIK pipeline test with a `SegmentClass` instance or faithful double.

## Completion Gate

The uv environments reproduce independently; `SegmentClass` remains compatible; preprocessing, training, evaluation, and inference commands work in both supported invocation forms; CPU tests and Linux CUDA smoke tests pass.
