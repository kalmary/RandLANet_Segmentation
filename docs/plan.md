# Semantic Segmentation Import Repair Plan

**Goal:** Separate inference imports from standalone preparation, training,
evaluation, and plotting imports while preserving the existing RandLANet
implementation and every supported command.

**Root-facing contract:** `SegmentClass(...).segment_pcd(points, intensity)`
retains its current validation, chunking, scaling, device behavior, point
ordering, and output labels.

**Parent plan:** `../../../docs/plan.md`

## Frozen behavior

- Do not split, rewrite, optimize, or retune semantic segmentation,
  voxel/chunk construction, overlap merging, scaling, or model prediction.
- Do not change model/config lookup conventions, CLI options, JSON schemas,
  dataset formats, output naming, or training/evaluation results.
- Preserve the nested `nn_utils` revision and API unless a separately approved
  canonical update is synchronized into every checkout.

## Completed foundation

- [x] Establish the Python 3.12 uv project and dependency groups.
- [x] Configure supported PyTorch profiles and preserve the nested `nn_utils`.
- [x] Test imports from the standalone project and parent BRIK repository.
- [x] Test direct/module help for existing preprocessing, inference, training,
  and evaluation commands.

## Task 1: Complete import characterization

- [x] Protect `SegmentClass` construction and `segment_pcd` behavior with the
  remaining deterministic characterization cases.
- [x] Inventory every `__main__` guard without removing any executable.
- [x] Add blocked-import tests proving which unrelated preparation, training,
  evaluation, plotting, HDF5, Open3D, Optuna, and Torchinfo modules currently
  load during inference.
- [x] Assert every `--help` path avoids datasets, weights, CUDA initialization,
  output creation, and plotting backends.

## Task 2: Repair inference imports

- [x] Replace `sys.path` mutation, generic `utils` imports, and wildcard imports
  with explicit imports from the defining packages/modules.
- [x] Import only the `nn_utils`, model, scaler, cache, and point-processing
  names actually used by inference.
- [x] Resolve configs and weights through explicit or stable project-relative
  paths without changing existing basename conventions.
- [x] Keep `src/array_processing.py` and `src/main.py` as compatible public
  locations; do not alter the algorithmic bodies they expose.

## Task 3: Repair standalone workflow imports

- [x] Keep LAZ/HDF5 preparation dependencies local to preparation commands.
- [x] Keep training, Optuna, Torchinfo, metrics, reporting, and plotting imports
  local to training/evaluation commands.
- [x] Remove only confirmed unused imports in a file when repairing that file's
  invocation path.
- [x] Ensure every current command parses arguments and `--help` before loading
  data, models, plotting, or CUDA state.
- [x] Preserve both direct-script and module execution from this project root.

## Task 4: Verify dependency ownership

- [x] Run inference import and deterministic CPU behavior tests under `basic`
  with the selected PyTorch extra.
- [x] Run all standalone tool invocation tests and the full suite under `dev`.
- [x] Verify parent-root import and root semantic stage behavior.
- [ ] Run production-weight CUDA verification separately on supported Linux
  hardware.
- [ ] Change dependency groups only after these workflow checks pass.

## Completion gate

Inference imports no offline or visualization workflows, every current tool
works independently in both declared forms, and `SegmentClass.segment_pcd`
produces unchanged results.
