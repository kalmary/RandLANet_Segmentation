# Independent code review

Scope: current preprocessing, source splitting, saved KD-trees, iterable loader, training and evaluation entry points, dense inference, and CLI output. This review was made from the current implementation rather than from `REVIEW.md` conclusions.

Verification performed: `python -m compileall -q src tests` completed successfully. After the recorded resolutions, the full unit suite completed with 34 tests passing.

## Critical

### 1. Single/grid training crashes at the first checkpoint

`case_based_training()` unpacks four values from `Checkpoint.check_checkpoint()` at `src/model_pipeline/TrainSegmAutomated.py:387`, but that method returns three values on both paths (`src/model_pipeline/TrainSegmAutomated.py:300` and `src/model_pipeline/TrainSegmAutomated.py:343`). The first yielded training epoch therefore raises `ValueError: not enough values to unpack`, so modes 1 and 2 cannot complete through the documented entry point.

Recommended fix: make the caller and return contract agree, and add a test that runs one mocked yielded epoch through `case_based_training()`.

Resolution: addressed after this review. `check_checkpoint()` consistently returns `(model, configuration, configuration_path)`, the single/grid caller unpacks those three values, and both checkpoint branches plus the first yielded epoch have regression coverage.

## High

### 2. The active pipeline still uses the model implementation that was meant to be replaced

Training, standalone evaluation, and dense inference import `RandLANet_CB`, not `RandLANet_CB_pooled`: `src/model_pipeline/_train_single_case.py:12`, `src/model_pipeline/TrainSegmAutomated.py:32`, `src/model_pipeline/EvalSegm_RandLANet.py:15`, and `src/array_processing.py:13-17`. Consequently, the pooled downsampling change in `RandLANet_CB_pooled.py:190-196` is not exercised by any normal pipeline path.

Recommended fix: perform the planned single-source model migration before relying on new training results, then test checkpoint compatibility explicitly.

Resolution: addressed after this review. All active training, evaluation, dense inference, and CLI paths import `RandLANet_CB_pooled`; the old implementation is retained only as an unused source file.

### 3. Evaluation measures neighborhood occurrences, not unique points

The loader marks coverage with `seen`, but yields every point in every queried neighborhood (`src/model_pipeline/_data_loader.py:172-191`). With overlapping neighborhoods, a point can occur many times even when `max_seen=1`. Evaluation then flattens and appends every occurrence without a source identity (`src/model_pipeline/EvalSegm_RandLANet.py:58-69`). `MAX_POINTS_EVAL` samples those occurrences (`src/model_pipeline/EvalSegm_RandLANet.py:71-94`), so frequently repeated points have a larger probability of entering the assessment and a larger influence when the cap is `None`.

Recommended fix: return stable point indices for evaluation and aggregate logits once per source point, or define and document occurrence-weighted metrics as the intended statistic.

Resolution: addressed with `EvalSegm_Dense.py`. It runs `SegmentClass` on complete LAS/LAZ clouds and compares one hard prediction per original classified point. The legacy evaluator remains available but is no longer the preferred final assessment path.

### 4. `_mod` output can discard source extra dimensions

`main.py` calls `create_new_las(source_path, output_file)` with its default `light_format=True` at `src/main.py:148`. The light-header path removes every source extra dimension at `src/LAZ_utils.py:76-80` and recreates only the two tree fields at `src/LAZ_utils.py:81-84`. A file described as a modified copy can therefore silently lose unrelated extra-byte attributes.

Recommended fix: call the existing utility with `light_format=False` for CLI copies, or explicitly document that `_mod` is a reduced-format export. Add a test using a custom extra dimension.

Resolution: closed by the requested output contract. `_mod` files intentionally use the light format and preserve the required standard XYZ, intensity, and RGB dimensions when available; arbitrary extra dimensions are not required. Regression coverage verifies the required dimensions.

### 5. Optuna mode fails while loading its configuration

For mode 3, `load_config()` assigns `training_config['device']` directly, then logs an undefined local named `device` at `src/model_pipeline/TrainSegmAutomated.py:219-223`. This raises `NameError` before optimization begins.

Recommended fix: log the assigned value and add parser-to-configuration smoke tests for every mode.

## Medium

### 6. Preprocessing failures can be reported as successful source processing

`save_tiles()` catches a LAS loading/normalization error and returns normally at `src/data_processing/downsample_LAZ.py:148-152`. `split_dataset()` does not inspect a result and increments the source count unconditionally at `src/data_processing/downsample_LAZ.py:249-257`. A damaged or unreadable source can therefore leave an incomplete split while the final summary counts it as processed.

Recommended fix: let the exception propagate or return a saved-tile count/status that the caller verifies. Test a failing source among valid sources.

### 7. Positive split fractions can silently produce empty validation or test sets

Split sizes are floored with `int(...)` at `src/data_processing/downsample_LAZ.py:212-215`. Small source collections can get zero validation/test files even when their configured fractions are positive. Training or evaluation then fails later with a less direct missing-data error.

Recommended fix: validate minimum source counts or allocate positive splits with an explicit policy, then test boundary sizes.

### 8. Validation mIoU is an average of per-batch ratios

Validation computes mIoU independently for each sampled neighborhood batch and averages those values by batch size (`src/model_pipeline/_train_single_case.py:204-224`). IoU is not additive, so this generally differs from computing intersections and unions over the validation stream. Overlapping neighborhood occurrences add the same sampling bias described for standalone evaluation.

Recommended fix: maintain a streaming confusion matrix/intersection-union accumulator. This preserves on-run computation without storing all logits.

### 9. Saved point/tree pairs are not transactional

`save_tiles()` writes the `.npy` first and the `.pkl` second at `src/data_processing/downsample_LAZ.py:178-182`. Failure during tree serialization leaves an unmatched point file. Because exclusive creation is used, rerunning the source then fails on that stale `.npy` instead of repairing the pair.

Recommended fix: write both artifacts to temporary names, then rename them only after both writes succeed. Add an injected pickle-failure test.

## Low

### 10. Dense subsampling still allocates a source-index array that is discarded

The dtype is now minimized, which materially reduces memory, but `segment_pcd()` still allocates one index for every dense input point at `src/array_processing.py:361-367` and ignores the returned indices. For very large clouds this remains avoidable memory pressure.

Recommended fix: let voxel subsampling return representative indices without requiring a label-sized payload, or add a path that omits the unused third array.

### 11. Zero-based model labels are shifted only at the LAS boundary

`src/main.py:143-150` adds one before writing classifications because LAS classification zero conventionally means unclassified. This is internally consistent and is only a format-boundary convention, but consumers must know that stored LAS classes are model labels plus one.

Recommended action: document the mapping next to the CLI output contract; no model-side change is required.

## Confirmed addressed areas

- Raw source paths are assigned to train/validation/test before tile creation (`src/data_processing/downsample_LAZ.py:192-260`), preventing tiles from one source crossing splits in a single run.
- Non-empty split directories are rejected (`src/data_processing/downsample_LAZ.py:242-246`), preventing stale tiles from a prior seed being mixed into a new split.
- Intensity preprocessing uses `MinMaxScaler` and handles constant values (`src/data_processing/downsample_LAZ.py:27-33`; `src/array_processing.py:354-359`).
- Saved cloud/tree names share a collision-resistant source-derived stem and use exclusive creation (`src/data_processing/downsample_LAZ.py:139-143`, `src/data_processing/downsample_LAZ.py:167-182`).
- Dense inference resets a defined random seed per cloud (`src/array_processing.py:32`, `src/array_processing.py:348`) and uses the smallest scalar index dtype (`src/array_processing.py:361-362`).
- Standalone evaluation is unweighted and uses `max_seen=1` (`src/model_pipeline/EvalSegm_RandLANet.py:40-48`); focal loss has no alpha weighting (`src/model_pipeline/EvalSegm_RandLANet.py:112-118`).
- CLI output names use `_mod`, avoid processing existing `_mod` inputs, preserve relative subdirectories, and reject an existing destination (`src/main.py:74-98`, `src/main.py:122-151`).

## Test gaps

- The checkpoint call contract has mocked first-yield coverage, but no end-to-end smoke test covers complete automated training modes.
- Dense evaluation verifies one contribution per classified source point; no end-to-end evaluation test uses a real checkpoint.
- No CLI output test verifies preservation of arbitrary LAS extra dimensions.
- No preprocessing test injects a read/serialization failure or checks split rounding for small source counts.
- Model tests do not assert that all entry points instantiate the intended pooled implementation.
