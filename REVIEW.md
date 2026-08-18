# Code review findings

Reviewed: 2026-07-31

## Critical

1. `src/main.py` was not executable end to end:
   - the argument parser was not called
   - dictionaries were passed positionally to `**kwargs` functions
   - an empty output-path string received `.mkdir()`
   - CUDA availability checked the function object instead of calling it
   - verbose file counting exhausted the input generator

## High priority

2. `src/model_pipeline/_train_single_case.py` hardcodes the loss device to CUDA independently of the configured training device.

3. Training validation computes weighted accuracy and mIoU per batch and averages them. Batch-averaged mIoU is not global mIoU, and weighted accuracy is averaged by batch size rather than total sample weight.

4. OneCycleLR uses the length of one stochastic coverage pass, while later epochs may contain a different number of batches.

5. Standalone evaluation aggregates sampled neighborhood occurrences rather than one finalized prediction per unique cloud point. Rare classes are sampled more often and then weighted again in weighted accuracy.

6. Training reports weighted cross-entropy loss, while standalone evaluation reports `FocalLoss_ArcFace`; the values are not comparable.

7. Inference returns zero-based labels and `src/main.py` wrote them directly to LAS classification, where zero means never classified.

8. `downsample_LAZ.load_and_normalise()` divides by zero when every intensity is zero, producing NaN features.

9. Reusing a dataset split output directory with another seed can leave a scan in multiple splits because old assignments are not removed.

## Medium priority

10. Evaluation retains every logits batch and then concatenates them, temporarily keeping two complete copies. Probabilities and plotting arrays increase the peak further.

11. Dense inference allocates an `int64` source-index array for every original point even though the returned source indices are discarded.

12. Dense inference uses an unseeded random generator, so identical clouds can produce different neighborhood contexts and potentially different labels.

## Test gaps

- No training-loop test.
- Evaluation tests mock the real loader.
- Array-processing tests bypass configuration and checkpoint loading.
- No coverage for zero intensity, repeated splitting, or a real saved model.

## Resolution status

- Finding 1: addressed by the argument-driven `src/main.py` processing path.
- Finding 2: closed by design. Training and standalone evaluation are CUDA-only, and their parsers no longer advertise CPU/GPU aliases.
- Finding 3: accepted for now. Validation metrics remain on-the-run batch aggregates.
- Finding 4: addressed pragmatically. Weighted targets are fixed, but random neighborhood overlap makes epoch batch counts variable. OneCycleLR now advances only while its configured step budget remains and stays at its final learning rate for extra batches.
- Finding 7: addressed by restoring one-based LAS classification before writing modified files.
- Command-line coverage now includes parsing, direct-file output, recursive directory output, relative-path preservation, LAS label restoration, and unavailable-CUDA rejection.
- Review work resumes from finding 5 when requested.
