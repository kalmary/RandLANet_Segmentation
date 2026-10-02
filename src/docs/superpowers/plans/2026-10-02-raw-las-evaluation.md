# Raw LAS/LAZ Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild evaluation so it segments complete raw LAS/LAZ clouds and computes sampled classification statistics from at most 50,000 stratified labeled points per file.

**Architecture:** The existing `SegmentClass.segment_pcd` inference path remains unchanged. The evaluator discovers raw clouds, performs one full-cloud inference per file, writes a deterministic post-inference stratified target/prediction sample to a temporary `.npy` file, then aggregates those samples into accuracy, mIoU, per-class IoU, and a text classification report.

**Tech Stack:** Python 3.12, argparse, pathlib, tempfile, NumPy, laspy, PyTorch, scikit-learn metrics through the existing `ClassificationReport`, pytest.

**Spec:** `src/docs/superpowers/specs/2026-10-02-raw-las-evaluation-design.md`

## Global Constraints

- `SegmentClass.segment_pcd` and all inference behavior below it must not change.
- Every discovered cloud is passed to `segment_pcd` in full exactly once, before metric sampling.
- Model predictions remain zero-based; persisted ground truth `1..N` is decoded to `0..N-1`, and persisted `0` is excluded from metrics.
- `--max_points` defaults to exactly `50000` and caps each file independently.
- Mode `0` performs validation only; mode `1` performs inference and statistics.
- Mode `1` must remove all temporary sample files after success or failure.
- Statistics are accuracy, mIoU, IoU per class, and a text classification report; no confusion matrix and no loss.

## Review Focus

- A cloud containing stored class `0` alongside labeled points must still be inferred in full while only `1..N` contributes to metrics; pin this in Task 3.
- Singleton/rare classes must not make stratification fail and must receive a sample when the budget permits; pin this in Task 2.
- A requested CUDA device on a machine without CUDA must fail instead of silently using CPU; pin this in Task 3.
- An exception during the second or later file must leave no temporary directory or `.npy` samples; pin this in Task 4.
- Predictions or decoded targets outside `0..num_classes-1` must identify the offending source file; pin this in Task 3.

---

### Task 1: Let `SegmentClass` load split training artifacts

**Files:**
- Modify: `src/array_processing.py`
- Test: `src/array_processing.py`

**Interfaces:**
- Consumes: existing `SegmentClass(..., config_dir=...)` calls.
- Produces: `SegmentClass(..., config_dir: str | Path, model_dir: str | Path | None = None)`, `SegmentClass.config -> dict`, and `SegmentClass.n_classes -> int`; omitted `model_dir` preserves the current same-directory behavior.

- [ ] **Step 1: Write the failing constructor test**

Add `test_segment_class_accepts_separate_config_and_model_directories` using monkeypatched `_load_config` and `_load_segmModel`. Assert that the constructor passes the explicit config directory only to config loading, the explicit model directory only to model loading, and exposes the loaded top-level `num_classes` through `n_classes`. Add a legacy assertion that omitting `model_dir` passes `config_dir` to both loaders.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest src/array_processing.py::test_segment_class_accepts_separate_config_and_model_directories -v`

Expected: FAIL because `model_dir` and `n_classes` do not exist.

- [ ] **Step 3: Implement the minimal constructor extension**

Add the optional `model_dir` after `config_dir`, resolve both paths relative to `array_processing.py` when they are relative, call `_load_config(config_dir)` and `_load_segmModel(model_dir or config_dir)`, and add read-only `config` and `n_classes` properties. Do not edit `segment_pcd` or any lower inference method.

- [ ] **Step 4: Run the constructor and existing segmentation tests**

Run: `pytest src/array_processing.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/array_processing.py
git commit -m "feat: load segmentation artifacts from separate directories"
```

### Task 2: Add deterministic stratified sampling and metric calculation

**Files:**
- Create: `src/model_pipeline/test_evaluation.py`
- Modify: `src/model_pipeline/EvalSegm_RandLANet.py`

**Interfaces:**
- Consumes: zero-based NumPy target and prediction arrays.
- Produces: `_stratified_indices(targets: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray` and `calculate_metrics(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> EvaluationMetrics`.

- [ ] **Step 1: Write failing sampling tests**

Add tests asserting that sampling returns every index below the cap; returns exactly the cap above it; preserves singleton classes when `max_points >= number_of_classes`; follows proportional largest-remainder allocation; contains no duplicates; and returns the same result for two generators seeded with `0`. Add validation tests for non-positive caps.

- [ ] **Step 2: Run the sampling tests to verify they fail**

Run: `pytest src/model_pipeline/test_evaluation.py -k stratified -v`

Expected: FAIL because `_stratified_indices` does not exist.

- [ ] **Step 3: Implement `_stratified_indices`**

Use `np.unique(..., return_counts=True)`, reserve one slot per class when possible, distribute remaining capacity proportionally with largest-remainder rounding while respecting each class population, sample each class without replacement using the supplied generator, combine the selected source indices, and sort them to preserve source order.

- [ ] **Step 4: Write failing metric tests**

Assert exact accuracy, mIoU, and ordered per-class IoU for a small known target/prediction pair. Assert mismatched, empty, negative, and out-of-range arrays raise descriptive `ValueError`s.

- [ ] **Step 5: Run metric tests to verify they fail**

Run: `pytest src/model_pipeline/test_evaluation.py -k metrics -v`

Expected: FAIL because the raw-evaluation metric interface is absent.

- [ ] **Step 6: Implement `calculate_metrics`**

Define an `EvaluationMetrics` `TypedDict`, calculate ordinary accuracy with NumPy, call existing `compute_mIoU` with int64 tensors, and return NumPy class IoUs plus flattened predictions and targets.

- [ ] **Step 7: Run Task 2 tests**

Run: `pytest src/model_pipeline/test_evaluation.py -k 'stratified or metrics' -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/model_pipeline/EvalSegm_RandLANet.py src/model_pipeline/test_evaluation.py
git commit -m "feat: add stratified evaluation sampling"
```

### Task 3: Build raw-cloud discovery and full-inference collection

**Files:**
- Modify: `src/model_pipeline/EvalSegm_RandLANet.py`
- Modify: `src/model_pipeline/test_evaluation.py`

**Interfaces:**
- Consumes: `SegmentClass.segment_pcd(points, intensity) -> np.ndarray`, LAS/LAZ files, `max_points`, a temporary directory, and an RNG.
- Produces: `_cloud_files(raw_path: Path) -> list[Path]`, `_device(name: str) -> torch.device`, and `collect_samples(segmenter: SegmentClass, files: list[Path], max_points: int, temp_dir: Path, rng: np.random.Generator, verbose: bool = True) -> CollectionSummary`. `CollectionSummary` is a `TypedDict` with `sample_paths: list[Path]`, `processed_files: int`, `labeled_files: int`, and `sampled_points: int`.

- [ ] **Step 1: Write failing discovery and device tests**

Test recursive, sorted, case-insensitive discovery of `.las` and `.laz`, rejection of a file path and an empty directory, CPU resolution, available CUDA resolution, and unavailable CUDA rejection.

- [ ] **Step 2: Run the discovery/device tests to verify they fail**

Run: `pytest src/model_pipeline/test_evaluation.py -k 'cloud_files or device' -v`

Expected: FAIL because the helpers do not exist.

- [ ] **Step 3: Implement discovery and exact device selection**

Use `Path.rglob('*')`, suffix normalization, sorted results, and descriptive errors. Permit only parser-approved `cpu` and `cuda`; never translate unavailable CUDA to CPU.

- [ ] **Step 4: Write failing collection tests**

Use fake laspy clouds and a recording segmenter. Assert each file supplies all coordinates and intensity to one `segment_pcd` call before sampling; stored labels `[0, 1, 2]` pair predictions only at stored labels `1, 2` and decode targets to `[0, 1]`; an entirely unlabeled file is still inferred but writes no sample; over-cap labels produce an exact stratified `.npy` sample; input arrays and prediction lengths are validated; and invalid target/prediction classes report the path.

- [ ] **Step 5: Run collection tests to verify they fail**

Run: `pytest src/model_pipeline/test_evaluation.py -k collect_samples -v`

Expected: FAIL because `collect_samples` does not exist.

- [ ] **Step 6: Implement full-cloud collection**

For each sorted file, read the cloud, materialize `x/y/z` and intensity, call `segment_pcd` once on all points, validate the returned one-dimensional shape and range, select candidates where stored classification is greater than zero, decode those targets by subtracting one, call `_stratified_indices`, and save one two-column `[target, prediction]` NumPy array as `sample_<file-index>.npy` in `temp_dir`. Return counts and paths; do not aggregate samples in memory inside the loop.

- [ ] **Step 7: Run Task 3 tests**

Run: `pytest src/model_pipeline/test_evaluation.py -k 'cloud_files or device or collect_samples' -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/model_pipeline/EvalSegm_RandLANet.py src/model_pipeline/test_evaluation.py
git commit -m "feat: evaluate complete LAS and LAZ clouds"
```

### Task 4: Add CLI orchestration, temporary cleanup, and reporting

**Files:**
- Modify: `src/model_pipeline/EvalSegm_RandLANet.py`
- Modify: `src/model_pipeline/test_evaluation.py`

**Interfaces:**
- Consumes: CLI arguments `--model_name`, `--raw_path`, `--device`, `--mode`, and `--max_points`; model artifacts in the existing training-results layout.
- Produces: `_model_paths(model_name: str) -> ArtifactPaths`, `parser(args: Sequence[str] | None = None) -> argparse.Namespace`, `run_dry_run(args: argparse.Namespace) -> None`, `run_evaluation(args: argparse.Namespace) -> EvaluationMetrics`, and `main() -> None`. `ArtifactPaths` is a `TypedDict` containing `model_dir`, `config_dir`, `model_path`, and `report_dir` paths.

- [ ] **Step 1: Write failing parser/path tests**

Assert required model/raw path arguments, defaults `device='cpu'`, `mode=0`, and `max_points=50000`, accepted explicit values, rejection of `.pt` in the model name, rejection of a non-directory raw path, and rejection of non-positive caps. Assert model `NAME_2` resolves to `training_results/NAME/{NAME_2.pt,dict_files/NAME_2_config.json}` and the existing `plots` directory target.

- [ ] **Step 2: Run parser/path tests to verify they fail**

Run: `pytest src/model_pipeline/test_evaluation.py -k 'parser or model_paths' -v`

Expected: FAIL against the old CLI.

- [ ] **Step 3: Implement parser, artifact resolution, and segmenter construction**

Replace the old evaluator imports and preprocessed-data model loader. Construct `SegmentClass(model_name=..., config_dir=dict_files, model_dir=model_dir, device=resolved_device, scaled=True, voxel_size_big=100.0, overlap=0.4, verbose=True)` and validate both artifact files before construction.

- [ ] **Step 4: Write failing dry-run and evaluation tests**

Assert mode `0` discovers files and constructs the segmenter but never calls `laspy.read` or `segment_pcd`. For mode `1`, monkeypatch collection/report dependencies and assert the report path/name, printed accuracy and IoUs, lack of any `Plotter`/confusion-matrix call, RNG seed behavior, and a descriptive no-labeled-samples failure.

Wrap a real temporary-directory spy around both a successful collection and a collection that raises on a later file; assert its path and all `.npy` children are absent after `run_evaluation` returns or raises.

- [ ] **Step 5: Run orchestration tests to verify they fail**

Run: `pytest src/model_pipeline/test_evaluation.py -k 'dry_run or run_evaluation or report' -v`

Expected: FAIL because the new orchestration is not implemented.

- [ ] **Step 6: Implement dry-run and mode-1 orchestration**

Create one `TemporaryDirectory(prefix='randlanet-evaluation-')` context in `run_evaluation`, one `default_rng(0)`, call `collect_samples`, load and concatenate its two-column arrays only after inference completes, calculate metrics, create the report directory, call `ClassificationReport`, print the metric/file/sample summary, and return metrics. Dispatch mode `0` and `1` in `main`.

- [ ] **Step 7: Run all evaluation tests**

Run: `pytest src/model_pipeline/test_evaluation.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/model_pipeline/EvalSegm_RandLANet.py src/model_pipeline/test_evaluation.py
git commit -m "feat: rebuild raw point-cloud evaluation CLI"
```

### Task 5: Verify invocation compatibility and complete regression coverage

**Files:**
- Modify if needed: `src/model_pipeline/EvalSegm_RandLANet.py`
- Modify if needed: `src/model_pipeline/test_evaluation.py`

**Interfaces:**
- Consumes: both `python src/model_pipeline/EvalSegm_RandLANet.py` and `python -m src.model_pipeline.EvalSegm_RandLANet` invocation forms.
- Produces: equivalent help and runtime imports in both forms.

- [ ] **Step 1: Run CLI help in both invocation forms**

Run: `python src/model_pipeline/EvalSegm_RandLANet.py --help` and `python -m src.model_pipeline.EvalSegm_RandLANet --help`

Expected: both exit `0` and list `--model_name`, `--raw_path`, `--device`, `--mode`, and `--max_points`.

- [ ] **Step 2: Run focused and project regression suites**

Run: `pytest src/model_pipeline/test_evaluation.py src/array_processing.py tests/test_invocation.py tests/test_imports.py -v`

Expected: PASS.

- [ ] **Step 3: Run static checks available in the repository**

Run: `python -m compileall -q src/model_pipeline/EvalSegm_RandLANet.py src/array_processing.py` and `git diff --check`

Expected: both exit `0` with no output from `git diff --check`.

- [ ] **Step 4: Review the final diff against the spec**

Confirm no HDF5 `Dataset`/`DataLoader`/loss path remains in evaluation, no confusion matrix is generated, `segment_pcd` is unchanged, sampling happens only after full inference, and temporary cleanup is context-managed.

- [ ] **Step 5: Commit any verification fixes**

```bash
git add src/model_pipeline/EvalSegm_RandLANet.py src/model_pipeline/test_evaluation.py src/array_processing.py
git commit -m "test: verify raw evaluation workflow"
```
