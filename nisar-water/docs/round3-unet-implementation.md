# Small NISAR U-Net implementation

This handoff describes code and synthetic software verification. It does not
claim a trained scientific model or measured real-data accuracy. Root owns the
observed dataset, actual GPU fit, family/threshold selection and reserved test.

## Runtime and ownership

The implementation is `src/nisar_water/neural.py`, with a thin command in
`scripts/train_unet_round3.py`. Use the existing CUDA environment at
`../nisar-models/.venv/Scripts/python.exe` and set `PYTHONPATH` to this package's
`src` directory. The ordinary Python3.13 water/API environment has no Torch.
No GPU packages or public pretrained weights were installed or downloaded.
Imports need only Python, NumPy and Torch; the model path does not require
SciPy, scikit-learn or segmentation-models-pytorch.

## Manifest

```json
{
  "data_origin": "observed",
  "label_kind": "weak",
  "scenes": [
    {"id": "scene-001", "path": "scenes/scene-001.npz", "split": "train", "group": "region-a", "region": "region-a", "acquired_at": "2026-10-01T00:00:00Z"},
    {"id": "scene-002", "path": "scenes/scene-002.npz", "split": "validation", "group": "region-b", "region": "region-b", "acquired_at": "2026-10-01T00:00:00Z"}
  ]
}
```

Each numeric NPZ contains `power` float32 `(2,H,W)` ordered HHHH/HVHV linear
gamma0, `valid` boolean `(H,W)`, `labels` integer `(H,W)` with 0 nonwater,
1 water and 255 unknown, and optional `weights` float32 `(H,W)` in [0,1].
`transform` (six affine values) and scalar `epsg` are retained when supplied.
An optional per-scene `sha256` is enforced. Paths may be manifest-relative.
Every group and region must occur in exactly one partition. Test scene metadata
can appear in the manifest, but test paths are never opened, stat'ed or hashed
by training. Root must additionally audit actual physical cell separation;
different region strings alone do not prove geographic independence.

Unknown, cloud and invalid labels never become negatives. Scenes with no known
valid labels do not supply supervised patches; they are recorded in the plan.
Observed training-scene pixels may still contribute to train-only normalization.
Synthetic training is rejected unless the explicit software-fixture override is
used, and such outputs retain `synthetic_fixture` provenance.

## Fixed fit

```python
from nisar_water.neural import train_unet
metadata = train_unet(
    manifest_path, new_output_directory,
    epochs=20, patch_size=128, patches_per_scene=16,
    batch_size=8, device='auto', seed=20261028,
)
```

The base24 three-pooling-level U-Net has **1,084,465 trainable parameters**.
It is fitted from scratch; it is not a pretrained Sentinel-1 or NISAR network.
Two convolution/BatchNorm/ReLU layers per block and learned stride2 upsampling
provide spatial context. No per-image or per-tile statistics normalize inputs.

Power undergoes exactly one `10log10`, fixed clipping to [-60,+10] dB and
standardization with training-scene-only mean/std. Normalization gives each
training scene equal contribution, sampling at most8,192 valid pixels per scene;
the standard-deviation floor is2dB. Invalid input is filled with normalized mean
zero, while supervised loss ignores invalid or unknown cells.

Each epoch has `patches_per_scene * number_of_train_scenes` patches. Geographic
groups receive an equal patch budget; a scene is selected within the group.
Where both labelled classes exist, patch centres are sampled with equal class
probability. This prevents dates from one region or a large all-water image
from dominating solely by pixel count. Validation preserves its natural support.
Augmentations are horizontal/vertical flips and multiples of90degrees only.

The loss is masked BCE plus0.5 masked soft-Dice. AdamW uses learning rate0.001
and weight decay0.0001; gradients are clipped at5. CUDA uses mixed precision.
DataLoader workers are0; no extra GPU jobs or background workers are started.

The best epoch minimizes mean validation-region loss. Each region contributes
the mean of its scored scenes, preventing extra validation dates from dominating.
The saved threshold remains0.5. `validation-thresholds.json` reports the declared
grid from0.1 through0.9 per scene, per region and pooled; root selects the final
family/operating point before opening new reserved test pixels.

## Saved artifacts and inference

`weights.pt` contains a tensor-only state dictionary. The loader requires
`torch.load(..., weights_only=True)` and verifies the recorded SHA256 and file
size. It never falls back to executable pickled model loading. `model.json`
declares `model_type: unet`, input contract, normalization, fitted epoch, label
provenance, checkpoint hash, software and implementation hash. The plan, epoch
history, validation threshold reports and per-scene validation score rasters
are preserved. Existing directories are never overwritten.

`NeuralWaterModel(directory, device='auto')` exposes `metadata`, `model_id`,
`fingerprint` and `predict(observation, threshold=None, tile_size=256)` compatible
with `run_series`. Scores are uncalibrated. Codes remain0/1/254/255. Source
invalidity and outside-AOI cells stay255; exclusions and unsupported observed
cells stay254. The same80% support requirement in3/7-cell neighborhoods as the
current local profile is retained for comparable eligibility.

Feature tiles overlap by128 pixels. A64-pixel receptive halo is discarded,
and tile starts align with the network's8-pixel pooling phase. BatchNorm uses
fixed running statistics in evaluation, so predictions do not change because
another tile contains different statistics. Crop size still affects context
available at the outer physical boundary; this is distinct from computational
tile seams and must be disclosed in scientific comparisons.

## Software verification

Seven tests in `tests/test_neural.py` passed in7.43seconds on CPU in the existing
Torch runtime. They check exactly-once power conversion, fixed training-only
normalization, zero loss gradients on unknown labels, no test-path reads,
geographic split rejection, group-balanced patch sampling, a one-epoch explicitly
synthetic CPU fixture fit, tensor checkpoint hashes, QA mask preservation and
tile-size invariance (32 versus256) across a nontrivial168×184 scene.
No GPU fitting or real-data accuracy evaluation was performed by this subagent.

