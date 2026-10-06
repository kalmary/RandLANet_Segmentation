# Raw LAS/LAZ Evaluation Design

## Goal

Replace evaluation against preprocessed HDF5 data with end-to-end evaluation of raw LAS/LAZ point clouds. Every source cloud must be segmented in full through the existing `SegmentClass.segment_pcd` inference method, while metrics use at most a stratified sample of 50,000 labeled points from each file by default.

## Command-line interface

`model_pipeline/EvalSegm_RandLANet.py` remains the evaluation entry point and accepts:

- `--model_name MODEL_NAME` (required): trained model name without `.pt`, resolved under the existing `model_pipeline/training_results/<base-name>/` layout.
- `--raw_path PATH` (required): directory searched recursively for `.las` and `.laz` files.
- `--device {cpu,cuda}` (default `cpu`): exact inference device. Requesting CUDA when CUDA is unavailable is an error rather than a silent CPU fallback.
- `--mode {0,1}` (default `0`): mode `0` is a dry run; mode `1` performs inference and writes statistics.
- `--max_points N` (default `50000`): maximum number of labeled points retained from each source file for metrics. It must be a positive integer.

Dry-run mode validates the arguments, source directory, discovered file set, model/config paths, requested device, and construction of the inference segmenter. It reports what would be evaluated but does not read complete clouds or call `segment_pcd`.

## Model loading

Evaluation must use `SegmentClass`, because that is the production raw-cloud inference pipeline. Its constructor will gain an optional model-directory argument so the trained `.pt` file and its JSON configuration can retain their current separate locations. Existing callers keep their current behavior when the argument is omitted. The implementation of `segment_pcd` and all inference behavior below it remain unchanged.

The evaluator obtains the class count from the loaded configuration and verifies that model predictions and decoded ground-truth labels are inside `0..num_classes-1`.

## File discovery and label convention

The raw path must be a directory. Discovery is recursive, case-insensitive for `.las` and `.laz`, deterministic by sorted path, and rejects an empty result.

For every discovered file, evaluation reads coordinates and intensity and calls `segment_pcd(points, intensity)` exactly once with the entire cloud. Model predictions already use zero-based class IDs and are never shifted.

LAS/LAZ classification values are the persisted representation: stored values `1..N` correspond to model classes `0..N-1`, so evaluation subtracts one from stored ground-truth values. Stored value `0` has no corresponding shifted class and is excluded from metric sampling. It is still included in full-cloud inference so it contributes spatial context. A file with no usable ground truth is fully segmented and then skipped for metric collection.

## Stratified metric sampling

Sampling occurs only after full-cloud inference. The candidate population is the set of points with usable ground truth, paired by source index with their predicted class.

If a file has no more than `max_points` candidates, all candidates are retained. Otherwise, an exact-size stratified sample is drawn without replacement. The budget is divided as evenly as possible among ground-truth classes present in the file. A class with fewer points than its quota contributes all its points, and the unused quota is redistributed evenly among classes that still have unsampled points until the budget is filled. A local `numpy.random.default_rng(0)` supplies the draws across the sorted file sequence, making repeated evaluations reproducible without changing NumPy's global random state. Sampling is stratified by ground truth, never by prediction.

## Temporary storage and aggregation

Mode `1` creates a unique system temporary directory using `tempfile.TemporaryDirectory`. Each file that contributes metrics writes its sampled targets and predictions as NumPy files. Keeping per-file samples on disk bounds evaluator memory during the inference loop and separates inference from final aggregation.

After all files finish, the evaluator loads the temporary samples, combines them, computes metrics, and writes the final report. The temporary directory is removed automatically on successful completion and on every exception. No temporary arrays remain in the source-data directory or repository.

If no file supplies usable labels, evaluation fails with a descriptive error after all files have been segmented.

## Statistics and output

The final sampled population is used to calculate:

- overall accuracy;
- mean intersection over union (mIoU);
- IoU for each model class;
- the existing text classification report, augmented with the accuracy and IoU values.

No confusion matrix is generated. Loss is not reported because production inference returns hard labels rather than training logits.

The text report is saved in the trained model's existing `plots` directory as `classification_report_<model_name>.txt`, preserving the current result location. A concise summary, including processed-file and sampled-point counts, is also printed.

## Failure behavior

Evaluation stops with a descriptive error for an unavailable requested device, missing model/config, unreadable or malformed cloud, mismatched input arrays, prediction length mismatch, or a ground-truth/prediction class outside the model range. The error identifies the source file where applicable. Temporary files are still removed.

## Testing

Automated tests will cover CLI defaults and validation, recursive LAS/LAZ discovery, dry-run behavior, separate model/config loading, full-cloud inference before sampling, stored-label offset handling, unlabeled-file behavior, exact stratified caps (including rare classes), deterministic sampling, metric values, report creation without a confusion matrix, and temporary-directory cleanup on success and failure.
