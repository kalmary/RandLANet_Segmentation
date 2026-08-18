# Repository status

Updated: 2026-07-31

## Working rules

- Do not run any Git operation without explicit user approval. Never commit or push.
- Preserve unrelated code and make surgical changes.
- Do not add generated-code or authorship/provenance comments.
- Follow the coding doctrine at <https://github.com/leopiney/linus-torvalds-skills>.
- Tests normally run through `.venv/bin/python` or `uv run`.

## Intended model direction

- `src/model_pipeline/RandLANet_CB_pooled.py` is the authoritative fixed model.
- Do not change how model-internal KNN is computed in `src/utils/knn_torch.py`.
- The project plans to replace the old RandLANet model with the pooled implementation later. This has not been done yet.
- `_train_single_case.py`, evaluation, and inference still import `RandLANet_CB.py` at present.

## Preprocessing

File: `src/data_processing/downsample_LAZ.py`

- Existing loading, normalization, voxel subsampling, tiling, filtering, and splitting logic was intentionally left unchanged.
- Valid LAS labels are filtered with `labels != 0` and shifted by one. This gives zero-based labels only when source classes are contiguous from 1 upward.
- Each tile remains an uncompressed float32 `.npy` array with shape `(N, 5)`:
  - columns 0-2: xyz
  - column 3: intensity
  - column 4: label
- Each tile now has a persisted SciPy `cKDTree` in a matching `.pkl` file.
- Tree configuration is `cKDTree(tile_xyz, leafsize=40)` and highest pickle protocol.
- Artifact names are:
  - `<scan>_<source-id>_tile_<i>_<j>.npy`
  - `<scan>_<source-id>_tile_<i>_<j>.pkl`
- The source identifier is a BLAKE2 digest of the resolved source path, distinguishing equal scan names from different directories.
- Files are opened with exclusive creation mode, so preprocessing refuses to overwrite an existing pair.

## Current sampling loader

File: `src/model_pipeline/_data_loader.py`

### Sampling behavior

- Loads one `.npy` tile and its exact-stem `.pkl` tree at a time.
- Validates `(N, 5)` data, `cKDTree` type, matching tree/point count, label range, and `N >= num_points`.
- Maintains one `seen` array per loaded tile with shape `(number_of_points, 1)` and dtype `int8`.
- `max_seen` must be from 1 through 127. Default is 10.
- Random centers are selected only from points below their class-specific target.
- Up to `batch_size` centers are selected together.
- SciPy finds neighborhoods in parallel with one call:

  ```python
  tree.query(centers, k=num_points, workers=query_workers)
  ```

- Every sample contains the selected center followed by `num_points - 1` nearest other points.
- Coordinates are centered on the selected point. Intensity and matching labels are gathered with the same indices.
- Every occurrence in a queried neighborhood increments `seen`. Updates use a wider temporary integer, saturate at 127, and cast back to `int8`, preventing overflow.
- Iteration stops only when every point reaches its target.

### Weight-to-coverage rule

- Training computes inverse-frequency weights directly from all labels in the saved `.npy` files.
- Those weights are already normalized to `[0, 1]`; the loader does not min-max normalize them again.
- The required number of views is proportional to the supplied weight:

  ```text
  target = max(1, round_half_up(weight * max_seen))
  ```

- Rarest class weight `1.0` receives `max_seen` views. A weight of `0.1` with `max_seen=10` receives one view.
- Supplied weights must be finite and within `[0, 1]`.

### Parallel delivery architecture

- `make_loader()` constructs a PyTorch `DataLoader` with:
  - `batch_size=None` because the dataset already yields complete batches
  - `num_workers=1`
  - `persistent_workers=True`
  - `prefetch_factor=2`
  - `pin_memory=False` to limit memory use
  - `multiprocessing_context="spawn"`
- The single persistent producer process owns the current cloud, tree, targets, and `seen` state.
- `query_workers`, an initialization parameter, controls SciPy KNN threads inside that producer. Training defaults to 7.
- Only prepared tensors cross PyTorch's multiprocessing queue; the dense cloud and tree are not copied to several DataLoader processes.
- Configuring more than one PyTorch worker is rejected.
- The persistent worker owns an internal iteration counter and advances its random seed every time a new loader iteration starts. Main-process `set_epoch()` was removed because it would not update the spawned persistent worker's dataset copy.

## Training integration

File: `src/model_pipeline/_train_single_case.py`

- Computes train and validation class weights before constructing weighted loaders.
- Uses `make_loader()` for both train and validation data.
- Passes `num_points`, `batch_size`, `query_workers`, `max_seen`, and weights.
- Datasets/loaders are constructed once and reused across epochs and repetitions.
- OneCycleLR step budget now includes `train_repeat`.
- The training configuration files contain:

  ```json
  "max_seen": 10,
  "query_workers": 7
  ```

- Validation no longer performs a separate loader pass only to count batches.
- Training still performs one initial weighted-loader pass because OneCycleLR needs a step count.
- Training is CUDA-only and its parser does not expose a device option.

### Known scheduler limitation

- Coverage-driven sampling can produce slightly different batch counts for different random seeds.
- OneCycleLR currently uses the first weighted pass as the fixed per-epoch estimate.
- OneCycleLR advances only while `last_epoch < total_steps`; extra stochastic batches keep the scheduler at its final learning rate without raising an exception.
- Removing the initial count pass or making scheduling exact requires selecting a scheduler compatible with variable-length epochs. This decision is still open.

## Evaluation

File: `src/model_pipeline/EvalSegm_RandLANet.py`

- Uses the current `make_loader()` interface and the saved point-cloud/tree pairs.
- Computes test class weights directly from the saved `.npy` files before constructing the coverage-driven test loader.
- The inference loop only collects CPU logits and labels.
- Loss, ordinary accuracy, weighted accuracy, mIoU, per-class IoU, probabilities, and predictions are computed once after the complete test loader is exhausted.
- The confusion matrix, precision-recall curve, ROC curve, and classification report all consume the same finalized predictions and labels.
- The classification report includes loss, accuracy, weighted accuracy, mIoU, and per-class IoU.
- Weighted accuracy uses supplied class weights directly. It no longer min-max normalizes them and therefore does not turn the smallest nonzero class weight into zero.
- Evaluation mode creates the plot directory when needed.
- Model construction and the evaluation smoke test use the current RandLANet constructor and output shape.
- Standalone evaluation is CUDA-only and its parser does not expose a device option.

## Dense-cloud inference

File: `src/array_processing.py`

- `SegmentClass` now takes inference preprocessing parameters directly: `voxel_size`, `tile_size`, `overlap`, `n_seen`, and `query_workers`.
- Dense XYZ is globally centered without modifying the caller's array. Intensity uses the same `log1p` normalization as preprocessing.
- The cloud is first reduced with `downsample_LAZ.voxel_subsample_vectorized()`.
- The subsampled cloud is split into regular overlapping XY tiles. The overlap is an absolute distance on every tile side.
- Every tile builds one SciPy `cKDTree` with `leafsize=40`.
- Each tile owns:
  - `seen`, shape `(tile_points, 1)`, dtype `int8`
  - accumulated probabilities, shape `(tile_points, n_classes)`, dtype `float32`
- Random centers are selected only while their own `seen` value is below `n_seen`. Already satisfied points remain eligible as neighbors of an unsatisfied center.
- Centers are queried in batches and SciPy uses `query_workers` internally.
- Parts smaller than the model point count are padded only for model input. Only real tree neighbors update probabilities and `seen`.
- Model logits are converted to probabilities. Each occurrence adds its probability vector and increments `seen`; final tile probabilities are divided by `seen` before `argmax`.
- Overlapping tile results use exact hard voting without an `(N, n_classes)` global array:
  - ordered labels have shape `(subsampled_points, max_overlaps)` and dtype `int8`
  - occurrence counts have shape `(subsampled_points,)` and dtype `int8`
  - a strict greater-than comparison preserves the first label when class vote counts tie
- For the default `tile_size=40` and `overlap=5`, a point has at most four tile votes, so global vote storage is five bytes per subsampled point.
- Final subsampled labels are transferred back to every original dense point with chunked nearest-neighbor queries.
- `src/main.py` now uses the renamed `tile_size` constructor argument where it creates `SegmentClass`.

## Command-line inference

File: `src/main.py`

- The command has one processing path and no mode/test switch.
- Supported arguments are:
  - required `--model_name`, without the `.pt` extension
  - optional `--verbose` flag
  - `--device` with `cpu` or `cuda`
  - required `--input_path`, accepting one `.las`/`.laz` file or a directory
  - optional `--output_path`, interpreted as an output directory
- CUDA requests fail clearly when CUDA is unavailable instead of silently falling back to CPU.
- A file input produces `<stem>_mod<suffix>`.
- Directory input recursively processes LAS/LAZ files and preserves their relative directory structure under `--output_path`.
- Existing `_mod` files are ignored during directory discovery.
- Without `--output_path`, each modified file is written beside its source.
- Existing output files are not overwritten.
- Zero-based model predictions are shifted back to one-based LAS classifications before writing.

## Tests

Files:

- `tests/test_downsample_LAZ.py`
- `tests/test_data_loader.py`
- `tests/test_evaluation.py`
- `tests/test_array_processing.py`
- `tests/test_main.py`

Coverage includes:

- distinct naming for equal source filenames
- exact `.npy`/`.pkl` pairing
- saved array content and dtype
- `cKDTree` serialization and queries
- overwrite prevention
- split propagation of paired artifacts
- direct weight-to-view targets
- complete per-point satisfaction
- `int8` target and seen arrays
- rejection of multiple DataLoader workers
- one persistent spawned producer
- disabled pinned memory
- preserved SciPy query-worker setting
- different sampling sequences across consecutive persistent-worker iterations
- inference collection before metric calculation
- global accuracy, weighted accuracy, mIoU, and per-class IoU
- direct use of supplied accuracy weights
- generation of all documented evaluation outputs from finalized arrays
- CUDA-only evaluation parser behavior
- exact hard voting with first-vote tie resolution
- per-occurrence probability and `seen` accumulation
- classification of every point in a small tile
- theoretical overlap bound and complete tile coverage
- hard voting across overlapping parts
- nearest-neighbor label transfer back to the original dense cloud
- command-line argument parsing
- direct-file and recursive-directory processing with real LAS files
- `_mod` output naming and relative-path preservation
- one-based LAS label restoration
- unavailable-CUDA rejection

Last verified result: 21 tests passed.

Run:

```bash
.venv/bin/python tests/test_data_loader.py
.venv/bin/python tests/test_downsample_LAZ.py
.venv/bin/python tests/test_evaluation.py
.venv/bin/python tests/test_array_processing.py
.venv/bin/python tests/test_main.py
```

The multiprocessing loader test requires permission to start PyTorch's shared-memory manager when run inside a restricted sandbox. It runs normally outside that sandbox.

## Not yet verified or completed

- No full training run has been performed because the configured datasets and target CUDA environment are not available locally.
- No full dense-cloud inference run has been performed with a real checkpoint and LAZ file.
- Model replacement/renaming to make the pooled model canonical has not been performed.
- The OneCycleLR variable-length epoch issue remains as described above.
