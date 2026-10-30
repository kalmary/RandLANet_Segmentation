# Density evaluation

Run the separate evaluator from each branch's repository root:

```bash
python src/model_pipeline/EvalDensity_RandLANet.py --device cuda --num_files 3
```

`DALES` evaluates `RandLANetDV0_2`; `prob-sampler-DALES` evaluates
`RandLANetPDV2_2`. Both select the first three sorted LAS/LAZ files from
`/mnt/data-ssd/DALES/raw/dales_las/test` by default. Modified `*_mod` outputs
are excluded. `--num_files` must be positive; if fewer files exist, all are used.

For each of **0.1, 0.25, 0.5, and 1.0 m**, the script independently subsamples
each original cloud. As in `prob-sampler`, it keeps the original point closest
to each occupied voxel's center, together with its intensity and label. The
voxel grid uses coordinates centered on the original cloud's mean.

Each branch's `SegmentClass` is imported and retains its current behavior.
DV0 intensity is scaled to `[0, 10]`, matching its training preprocessing;
PDV2 intensity normalization stays inside its inference pipeline. PDV2 can
subsample again internally; its returned indices keep ground truth aligned
with the actual predictions. Neither script reprojects labels onto the
original dense cloud.

Reports aggregate all evaluated labeled points across the selected files.
LAS classification 0 is retained as context but excluded from metrics;
other classifications become zero-based model labels. Each text file contains
the sklearn classification report (precision, recall, F1, support and aggregate
metrics), file paths, voxel size, seed and point counts. Reports are calculated
exactly from confusion counts, without a balanced sample or retaining all
predictions in memory. No CSV or additional metric artifacts are generated.

Default output files are:

```text
evaluation_results/density/<model>/classification_report_voxel_0.1.txt
evaluation_results/density/<model>/classification_report_voxel_0.25.txt
evaluation_results/density/<model>/classification_report_voxel_0.5.txt
evaluation_results/density/<model>/classification_report_voxel_1.txt
```

Optional flags:

- `--raw_path PATH`: raw test directory (searched recursively).
- `--output_dir PATH`: report root; a model subdirectory is added. Repeated runs overwrite that model's four reports.
- `--model_dir PATH`: checkpoint directory containing `<model>.pt` and `dict_files/<model>_config.json`.
- `--device cpu`: CPU inference (default); use `cuda` for GPU inference.
- `--seed 42`: sampling seed, reset per file consistently across voxel sizes.

Models and configs are read from their existing `training_results` directories
unless `--model_dir` is supplied. Reports are written separately.

Run both branches from the same repository directory:

```bash
git switch prob-sampler-DALES
.venv/bin/python src/model_pipeline/EvalDensity_RandLANet.py --device cuda --num_files 3

git switch DALES
.venv/bin/python src/model_pipeline/EvalDensity_RandLANet.py --device cuda --num_files 3
```

Each branch selects its own model automatically. Both checkpoints remain under
`src/model_pipeline/training_results/` when switching branches.
