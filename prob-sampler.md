# Probability-Sampler RandLA-Net Pipeline

## Scope

This document describes the probability-sampler pipeline shared by the
`prob-sampler` and `prob-sampler-DALES` branches. The learning, sampling,
model, loss, validation, checkpoint, and dense-inference logic is common to
both branches. Dataset-specific paths and raw-file organization are the main
differences:

- `prob-sampler` uses the BRIK dataset.
- `prob-sampler-DALES` uses the DALES dataset.

The pipeline performs semantic segmentation of point clouds. Training reads
preprocessed spatial tiles, while final assessment and production inference
read complete `.las` or `.laz` clouds through `array_processing.py`.

## Pipeline overview

The complete data flow is:

1. Read a raw LAS/LAZ scan.
2. Normalize intensity once over the complete scan.
3. Voxel-subsample the scan at 0.10 m.
4. Divide the scan into overlapping 60 m × 60 m spatial tiles.
5. Save each tile as an `(N, 5)` NumPy array and a matching KD-tree.
6. Split tiles by source scan into training, validation, and test groups.
7. Compute class-frequency weights from the training and validation tiles.
8. Build fixed-length, scan-balanced training and validation epochs.
9. Select crop centers using a mixture of natural spatial coverage and
   class-aware sampling.
10. Query 8,192-point neighborhoods around the selected centers.
11. Train RandLA-Net with normalized weighted focal loss.
12. Select checkpoints using global validation mean IoU.
13. Run final inference on complete raw LAS/LAZ files.
14. Compute exact metrics over every labeled point in each raw cloud.

## 1. Raw scan preparation

Preprocessing is implemented in `src/data_processing/downsample_LAZ.py`.

### Coordinates

LAS coordinates are first read as `float64`. The complete scan mean is
subtracted before converting coordinates to `float32`; this avoids losing
local precision when LAS coordinates contain large map offsets.

Each crop used for training is centered again on its selected center point.
The network therefore receives local metric XYZ offsets rather than absolute
map coordinates.

### Intensity

Intensity is normalized once for the complete source scan:

```text
normalized_intensity = intensity / scan_maximum_intensity
```

An all-zero scan remains all zero. Intensity is not normalized independently
per tile or per crop. Every neighborhood from the same source scan therefore
uses the same intensity scale.

The model input normalization preserves XYZ in metres and shifts intensity
from `[0, 1]` to approximately `[-0.5, 0.5]`.

### Labels and unclassified context

Positive LAS classification values are converted to zero-based model labels:

```text
model_label = las_classification - 1
```

When unclassified points are retained, LAS class `0` becomes label `-1`.
These points remain available as geometric and intensity context, but they:

- cannot become supervised crop centers;
- do not contribute to class-frequency weights;
- do not contribute to the focal loss;
- do not contribute to validation or evaluation metrics.

Existing tile sets without `-1` labels remain valid inputs to the same
loader.

## 2. Voxel subsampling

The configured voxel size is 0.10 m. For every occupied voxel, preprocessing
keeps the source point closest to the voxel center. Its XYZ, normalized
intensity, and label are retained together.

The same 0.10 m voxel size is recorded in the saved training configuration
and is used by dense inference unless an explicit inference override is
provided.

## 3. Spatial tiling

The point cloud is divided on the XY plane using a regular 60 m grid. Each
tile includes 5 m of overlap around its core spatial extent. The grid spacing
remains 60 m; overlap supplies boundary context to adjacent tiles.

Every nonempty tile is retained. A tile is not rejected because it is small,
contains few classes, or contains only frequent classes. Dataset imbalance is
handled by the sampler and the loss rather than by deleting spatial regions.

Each saved tile has two files with the same stem:

```text
<scan>_<source-id>_tile_<x-index>_<y-index>.npy
<scan>_<source-id>_tile_<x-index>_<y-index>.pkl
```

The `.npy` file has five columns:

| Column | Content |
|---:|---|
| 0 | X coordinate |
| 1 | Y coordinate |
| 2 | Z coordinate |
| 3 | globally normalized intensity |
| 4 | zero-based class label or `-1` context label |

The `.pkl` file contains a SciPy `cKDTree` built from the tile XYZ values.
The loader validates that the number of tree entries equals the number of
points in the NumPy array.

Dataset splitting is source-scan aware. Tiles from one source scan stay in a
single split, preventing tiles from the same scan from appearing in both
training and validation data.

## 4. Class weights

Before model training, the pipeline scans column 4 of every tile and counts
all non-negative labels. For class `c`, the unnormalized weight is:

```text
weight[c] = (1 / (count[c] + 1e-6)) ** class_weight_power
```

The configured `class_weight_power` is `0.5`, which applies inverse square-root
frequency weighting. Present-class weights are divided by their maximum, so
the rarest present class has weight `1.0`. A class absent from the split has
weight `0.0`.

The weights serve two related purposes:

1. They control the class-aware portion of crop-center selection.
2. They set relative per-class contributions in the focal loss.

Training and validation weights are computed independently from their
respective splits.

## 5. Fixed-length epochs

`CustomDataset` is an iterable dataset with an explicit batch budget:

```text
train_steps_per_epoch = 500
validation_steps      = 100
```

One training epoch therefore contains exactly 500 optimizer steps, and one
validation epoch contains exactly 100 batches. With 100 epochs, the optimizer
receives 50,000 training steps.

The fixed budget gives each progress bar an exact total and ETA. An epoch does
not mean exhaustive traversal of every point in every overlapping tile.

The PyTorch loader uses one persistent producer worker. KD-tree queries use
the separate `query_workers` setting:

- BRIK: 12 query workers;
- DALES: 7 query workers.

## 6. Scan-balanced tile scheduling

Tile filenames are grouped by the portion before `_tile_`, which identifies
their source scan. Scheduling proceeds in scan-balanced cycles:

1. Visit every source scan with equal frequency.
2. Select the next tile for that scan using a rotating cursor.
3. Produce up to `batches_per_file = 8` batches from the loaded tile.
4. Move to the next scan.
5. Continue until the epoch batch budget is reached.

Using several batches from one loaded tile amortizes the cost of loading a
large NumPy array and its KD-tree. Rotating tile cursors ensure that subsequent
visits to a scan progress through its tiles rather than repeatedly choosing
the first tile.

Training scan order is randomized per epoch. Validation uses a fixed random
seed and local tile cursors, producing the same crop sequence each epoch for
stable checkpoint comparisons.

## 7. Per-point coverage targets

Every labeled point receives a coverage target. With `max_seen = 3`, the
target for a point of class `c` is:

```text
target[c] = max(1, round(class_weight[c] * 3))
```

The minimum target is always one, including for very frequent classes with a
small class weight. Frequent classes therefore remain eligible for natural
center selection. Context-only points with label `-1` receive target zero and
cannot be selected as supervised centers.

For each loaded tile, the sampler initializes:

- a small random possibility value for every point;
- a `seen` counter for every point;
- the class-derived target for every labeled point.

The normalized coverage priority of a point is:

```text
seen[point] / target[point]
```

Points with lower normalized coverage are preferred.

## 8. Mixed crop-center selection

The configured `class_aware_center_fraction` is `0.5`. For a batch of six
crops, center selection normally consists of:

- three natural low-coverage centers;
- three class-aware low-coverage centers.

### Natural centers

Natural centers are selected by lowest normalized coverage and then lowest
possibility. This component follows the spatial and class distribution of the
scan, ensuring that frequent classes and ordinary background geometry remain
well represented.

### Class-aware centers

For each class-aware slot:

1. Determine which classes still have eligible points in the tile.
2. Draw a target class with probability proportional to its class weight.
3. Within that class, select the point with the lowest normalized coverage
   and then the lowest possibility.

This component increases access to rare classes without making the complete
batch class-balanced or excluding frequent classes.

## 9. Neighborhood construction

For every selected center, the tile KD-tree returns the nearest 8,192 real
points. The center is explicitly placed at index zero.

XYZ is converted to crop-relative metric coordinates:

```text
crop_xyz = neighbor_xyz - center_xyz
```

The normalized intensity column is concatenated without crop-level rescaling,
producing model input of shape:

```text
(batch_size, 8192, 4)
```

The corresponding label tensor has shape:

```text
(batch_size, 8192)
```

If a tile contains fewer than 8,192 points, real neighbors are repeated only
to produce a fixed-size model input. Coverage and possibility updates use only
the real neighbor set, so padding does not pretend that duplicate entries are
new spatial observations.

## 10. Possibility update

After querying a neighborhood, squared distances to the crop center are
normalized by the largest squared distance in that neighborhood. Each real
neighbor receives the update:

```text
delta = (1 - normalized_squared_distance) ** 2
possibility += delta
seen += 1
```

Points close to a selected center receive the largest possibility increase.
Subsequent selections therefore move toward spatial areas with lower previous
coverage.

Sampling from a tile stops when either:

- its requested batch block has been produced; or
- all labeled points in that tile have met their coverage targets.

The scan-balanced scheduler then selects another tile until the fixed epoch
budget is complete.

## 11. Model input and hierarchy

The model receives crop-relative XYZ and scan-normalized intensity. Model-side
input normalization keeps XYZ in metric units and subtracts `0.5` from the
intensity channel.

Every forward pass independently permutes the points of each batch element.
The encoder hierarchy takes progressively decimated prefixes from this random
ordering, making hierarchical subsampling spatially unbiased. Output logits
are restored to the original crop order using the inverse permutation.

The network uses:

- an initial shared MLP;
- RandLA-Net local spatial encoding;
- attentive local feature aggregation;
- residual encoder blocks;
- random-sampling aggregation along the encoder hierarchy;
- nearest-neighbor feature propagation in the decoder;
- skip connections from matching encoder resolutions;
- shared-MLP classification layers;
- dropout probability `0.4` in the single-model configuration.

The single-model configuration uses four encoder and four decoder stages,
16 nearest neighbors for local aggregation, and a decimation factor of 2.
The output shape is `(B, num_classes, N)`.

## 12. Supervised targets

Before loss calculation, logits are moved to channel-last order and flattened:

```text
logits:  (B, C, N) -> (B*N, C)
targets: (B, N)    -> (B*N)
```

Entries with target `-1` are removed from both tensors. At least one labeled
point must remain in every sampled batch.

## 13. Normalized weighted focal loss

The loss uses:

- focal gamma `1.0`;
- label smoothing `0.1`;
- split-specific inverse-frequency class weights.

For every labeled point, the base focal term is:

```text
focal_loss_i = (1 - p_true_i) ** gamma * smoothed_cross_entropy_i
```

Class weighting and reduction are:

```text
loss = sum(alpha[target_i] * focal_loss_i) / sum(alpha[target_i])
```

Dividing by the selected target weights preserves relative rare/frequent class
importance while keeping the absolute loss scale independent of a global
rescaling of `alpha`. Random ten-class predictions therefore produce a loss
on the ordinary cross-entropy scale instead of an artificially tiny value.

## 14. Optimization schedule

The single-training configuration uses:

| Setting | Value |
|---|---:|
| Training repetitions | 1 |
| Epochs | 100 |
| Training steps per epoch | 500 |
| Validation steps per epoch | 100 |
| Batch size | 6 crops |
| Points per crop | 8,192 |
| AdamW learning rate | 0.001 |
| AdamW weight decay | 0.01 |
| OneCycle warm-up fraction | 0.2 |
| OneCycle division factor | 100 |
| OneCycle final division factor | 10,000 |

The known epoch length gives OneCycleLR exactly 50,000 total optimizer steps.
The scheduler advances after every training batch.

## 15. Validation metrics

Validation uses the same model input representation and crop construction but
uses deterministic natural possibility sampling. Class-aware center selection
is disabled for validation so the crop distribution is not deliberately
reweighted by training class weights.

For every validation batch, predictions and targets update one global
`num_classes × num_classes` confusion matrix. At the end of the validation
epoch:

- accuracy is the diagonal sum divided by all labeled validation points;
- per-class IoU is intersection divided by union;
- mean IoU is averaged over classes present in the validation targets.

Loss is accumulated by labeled-point count rather than by batch count.

## 16. Checkpoint selection and artifacts

Training yields the model and metric history after each epoch. The checkpoint
manager selects the model with the highest validation mean IoU.

For a model base name such as `RandLANetPV2`, artifacts are stored below:

```text
src/model_pipeline/training_results/RandLANetPV2/
```

The directory contains:

- the `.pt` model state;
- `dict_files/<checkpoint>_config.json`;
- loss, accuracy, and mIoU plots.

The saved JSON combines the training settings and model architecture. Dense
inference reads spatial preprocessing values such as voxel size, tile size,
overlap, and intensity convention from this file.

## 17. Running training

### BRIK

```bash
git switch prob-sampler
.venv/bin/python src/model_pipeline/TrainSegmAutomated.py \
  --model_name RandLANetPV2 \
  --mode 1
```

The single-training configuration reads:

```text
/mnt/data-ssd/BRIK/SEMANTIC_SEGM/prob/split/train
/mnt/data-ssd/BRIK/SEMANTIC_SEGM/prob/split/val
```

### DALES

```bash
git switch prob-sampler-DALES
.venv/bin/python src/model_pipeline/TrainSegmAutomated.py \
  --model_name RandLANetPDV2 \
  --mode 1
```

The single-training configuration reads:

```text
/mnt/data-ssd/DALES/dataset_prob/train
/mnt/data-ssd/DALES/dataset_prob/val
```

Mode `1` loads `config_train_single.json` and
`config_model_single.json`. Mode `2` runs the Optuna search configuration.
The `data_path_test` field is not consumed by the mode-1 training loop.

## 18. Dense raw-cloud inference

Final inference uses `SegmentClass` from `src/array_processing.py` and reads a
complete raw cloud:

1. Validate XYZ and intensity arrays.
2. Center the complete XYZ array for numerical precision.
3. Normalize intensity by the complete cloud maximum.
4. Voxel-subsample at the model's configured voxel size.
5. Divide the subsampled cloud into overlapping spatial tiles.
6. Build a KD-tree for each nonempty inference tile.
7. Use possibility sampling until the configured vote threshold is reached.
8. Accumulate softmax probabilities for every real point occurrence.
9. Average probabilities within each inference tile.
10. Resolve predictions from overlapping spatial tiles by label voting.
11. Upsample subsampled labels to every original dense point using one-nearest
    neighbor assignment.

Unclassified raw points remain in the cloud during inference and can influence
the geometric context seen by labeled points.

## 19. Raw LAS/LAZ evaluation

`src/model_pipeline/EvalSegm_RandLANet.py` constructs `SegmentClass` and sends
complete `.las` or `.laz` scans through the dense inference path. Final test
assessment does not use preprocessed test tiles.

Predictions are produced for every raw point. Metrics use only points whose
stored LAS classification is greater than zero. The evaluator accumulates an
exact confusion matrix over all such points and reports:

- exact labeled-point accuracy;
- exact balanced accuracy;
- exact mean IoU;
- exact IoU per class.

A bounded, class-balanced sample is retained only for the human-readable
precision/recall classification table. It does not determine the headline
accuracy or IoU values.

## 20. Configuration summary

The main shared probability-sampler settings are:

```json
{
  "voxel_size": 0.10,
  "tile_size": 60.0,
  "overlap": 5.0,
  "num_points": 8192,
  "batch_size": 6,
  "num_neighbors": 16,
  "max_seen": 3,
  "class_weight_power": 0.5,
  "class_aware_center_fraction": 0.5,
  "batches_per_file": 8,
  "train_steps_per_epoch": 500,
  "validation_steps": 100,
  "epochs": 100,
  "focal_loss_gamma": 1.0,
  "learning_rate": 0.001,
  "weight_decay": 0.01,
  "intensity_normalization": "linear_max"
}
```

Together, these settings define a fixed-compute training process that retains
natural frequent-class exposure, increases rare-class center access, preserves
metric geometry, and keeps final assessment on complete raw point clouds.
