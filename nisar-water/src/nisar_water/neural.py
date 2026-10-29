"""Small NISAR HH/HV U-Net with masked supervision and bounded inference.

Torch is deliberately isolated from the ordinary CPU/API environment. Import this
module in the existing CUDA runtime; no pretrained weights are downloaded. Input
normalization is fitted from training scenes only and never from a test scene.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .common import read_json, write_json, sha256, utcnow

SCHEMA = "nisar-water-neural-v1"
FEATURE_VERSION = "nisar-l-hhhv-unet-20m-v1"
CHANNELS = ["HHHH", "HVHV"]
INFERENCE_HALO = 64
STRIDE = 8
MAX_SCENE_PIXELS = 16_000_000


class DoubleConv(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                         nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
                         nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                         nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))


class SmallUNet(nn.Module):
    """Three pooling levels; BatchNorm uses frozen running statistics at inference."""
    def __init__(self, base_channels=24):
        super().__init__()
        if not isinstance(base_channels, int) or not 4 <= base_channels <= 32:
            raise ValueError("base_channels must be an integer in [4,32]")
        b = base_channels
        self.enc0, self.enc1, self.enc2 = DoubleConv(2, b), DoubleConv(b, 2*b), DoubleConv(2*b, 4*b)
        self.bottom = DoubleConv(4*b, 8*b)
        self.up2, self.dec2 = nn.ConvTranspose2d(8*b, 4*b, 2, stride=2), DoubleConv(8*b, 4*b)
        self.up1, self.dec1 = nn.ConvTranspose2d(4*b, 2*b, 2, stride=2), DoubleConv(4*b, 2*b)
        self.up0, self.dec0 = nn.ConvTranspose2d(2*b, b, 2, stride=2), DoubleConv(2*b, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        height, width = x.shape[-2:]
        x = F.pad(x, (0, (-width) % STRIDE, 0, (-height) % STRIDE))
        e0 = self.enc0(x)
        e1 = self.enc1(F.max_pool2d(e0, 2))
        e2 = self.enc2(F.max_pool2d(e1, 2))
        x = self.bottom(F.max_pool2d(e2, 2))
        x = self.dec2(torch.cat([self.up2(x), e2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), e1], dim=1))
        x = self.dec0(torch.cat([self.up0(x), e0], dim=1))
        return self.head(x)[..., :height, :width]


def choose_device(device="auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return result


def _validate_arrays(power, valid, labels=None, weights=None):
    power, valid = np.asarray(power, np.float32), np.asarray(valid)
    if valid.ndim != 2 or power.shape != (2, *valid.shape) or valid.size > MAX_SCENE_PIXELS:
        raise ValueError("Expected bounded ordered HH/HV power (2,H,W) and valid (H,W)")
    if valid.dtype != np.bool_:
        raise ValueError("Scene validity must be boolean, not a native categorical mask")
    valid = valid & np.isfinite(power).all(0) & (power > 0).all(0)
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != valid.shape or not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, [0, 1, 255]).all():
            raise ValueError("Labels must be integer 0 land / 1 water / 255 unknown on the input grid")
        labels = labels.astype(np.uint8, copy=False)
    if weights is None:
        weights = np.ones(valid.shape, np.float32)
    weights = np.asarray(weights, np.float32)
    if weights.shape != valid.shape or not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
        raise ValueError("Weights must be finite [0,1] values matching the scene")
    return power, valid, labels, weights


def load_scene(spec, manifest_parent):
    path = Path(spec["path"])
    if not path.is_absolute():
        path = Path(manifest_parent) / path
    if spec.get("sha256") and sha256(path) != spec["sha256"]:
        raise ValueError("Scene archive hash differs from its manifest")
    with np.load(path, allow_pickle=False) as archive:
        power, valid, labels, weights = _validate_arrays(
            archive["power"], archive["valid"], archive["labels"],
            archive["weights"] if "weights" in archive else None)
        transform = archive["transform"].copy() if "transform" in archive else None
        epsg = int(archive["epsg"]) if "epsg" in archive else None
    return {**spec, "resolved_path": str(path.resolve()), "sha256": sha256(path),
            "power": power, "valid": valid, "labels": labels, "weights": weights,
            "transform": transform, "epsg": epsg}


def development_scenes(manifest_path, *, allow_synthetic_fixture=False):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    synthetic = manifest.get("data_origin") == "synthetic_fixture"
    if manifest.get("data_origin") != "observed" and not (synthetic and allow_synthetic_fixture):
        raise ValueError("Training requires observed data; synthetic fixtures need an explicit software-test override")
    if manifest.get("label_kind") not in ("weak", "reviewed", "mixed"):
        raise ValueError("Declare label_kind weak, reviewed or mixed")
    partitions = defaultdict(set)
    identifiers = set()
    for spec in manifest.get("scenes", []):
        for key in ("id", "path", "split", "group", "region"):
            if not spec.get(key):
                raise ValueError(f"Scene manifest needs {key}")
        if spec["id"] in identifiers:
            raise ValueError("Duplicate scene identifier")
        identifiers.add(spec["id"])
        if spec["split"] not in ("train", "validation", "test"):
            raise ValueError("Unsupported split")
        partitions[("group", spec["group"])].add(spec["split"])
        partitions[("region", spec["region"])].add(spec["split"])
    if any(len(splits) != 1 for splits in partitions.values()):
        raise ValueError("A geographic group or region occurs in multiple partitions")
    # Test paths are intentionally never opened, stat'ed or hashed here.
    loaded = {split: [load_scene(s, manifest_path.parent) for s in manifest["scenes"] if s["split"] == split]
              for split in ("train", "validation")}
    for split, scenes in loaded.items():
        known = sum(int((s["valid"] & (s["labels"] != 255) & (s["weights"] > 0)).sum()) for s in scenes)
        if not known:
            raise ValueError(f"No labelled valid support in {split}")
    return manifest, loaded


def fit_normalization(scenes, *, seed=20261028, samples_per_scene=8192):
    """Equal scene contribution; only training images supply normalization statistics."""
    rng = np.random.default_rng(seed)
    means, seconds = [], []
    for scene in scenes:
        indices = np.flatnonzero(scene["valid"])
        if not len(indices):
            continue
        if len(indices) > samples_per_scene:
            indices = rng.choice(indices, samples_per_scene, replace=False)
        values = scene["power"].reshape(2, -1)[:, indices]
        db = np.clip(10*np.log10(values), -60., 10.).astype(np.float64)
        means.append(db.mean(1))
        seconds.append((db*db).mean(1))
    if not means:
        raise ValueError("No valid training power for normalization")
    mean = np.mean(means, axis=0)
    std = np.maximum(np.sqrt(np.maximum(np.mean(seconds, axis=0)-mean**2, 0)), 2.)
    return {"mean_db": mean.tolist(), "std_db": std.tolist(), "clip_db": [-60., 10.],
            "fitted_on": "train_only", "weighting": "equal_scene_capped_valid_pixels"}


def normalize_power(power, valid, normalization):
    """Input is linear power; one 10log10 conversion, fixed training normalization."""
    values = np.asarray(power, np.float32)
    good = np.asarray(valid, bool) & np.isfinite(values).all(0) & (values > 0).all(0)
    mean = np.asarray(normalization["mean_db"], np.float32)[:, None, None]
    std = np.asarray(normalization["std_db"], np.float32)[:, None, None]
    if mean.shape != (2, 1, 1) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid fixed normalization")
    db = 10*np.log10(np.where(good[None], values, 1.))
    db = np.clip(db, *normalization["clip_db"])
    result = (db-mean)/std
    result[:, ~good] = 0
    return result.astype(np.float32, copy=False)


def classification_support(valid):
    """Same observed-support gate as the existing local feature profile."""
    valid = np.asarray(valid, bool)
    supported = valid.copy()
    for size in (3, 7):
        p = np.pad(valid.astype(np.int32), size//2)
        integral = np.pad(p, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        count = integral[size:, size:]-integral[:-size, size:]-integral[size:, :-size]+integral[:-size, :-size]
        supported &= count >= math.ceil(.8*size**2)
    return supported


def _predict_scores(model, power, valid, normalization, device, *, tile_size=256):
    """Phase-aligned input tiles overlap by 128 pixels; receptive halos are discarded."""
    if tile_size < STRIDE:
        raise ValueError("tile_size must be at least eight pixels")
    core = max(STRIDE, int(tile_size)//STRIDE*STRIDE)
    height, width = valid.shape
    score = np.full(valid.shape, np.nan, np.float32)
    model.eval()
    with torch.inference_mode():
        for row in range(0, height, core):
            for col in range(0, width, core):
                r1, c1 = min(row+core, height), min(col+core, width)
                r0 = max(0, (row-INFERENCE_HALO)//STRIDE*STRIDE)
                c0 = max(0, (col-INFERENCE_HALO)//STRIDE*STRIDE)
                endr, endc = min(height, r1+INFERENCE_HALO), min(width, c1+INFERENCE_HALO)
                block = normalize_power(power[:, r0:endr, c0:endc], valid[r0:endr, c0:endc], normalization)
                tensor = torch.from_numpy(np.ascontiguousarray(block[None])).to(device)
                probability = torch.sigmoid(model(tensor))[0, 0].float().cpu().numpy()
                if not np.isfinite(probability).all():
                    raise ValueError("Neural inference produced nonfinite scores")
                score[row:r1, col:c1] = probability[row-r0:r1-r0, col-c0:c1-c0]
    score[~valid] = np.nan
    return score


def masked_bce_dice(logits, target, weights):
    """Unknown/invalid cells have weight zero and contribute to neither term."""
    denominator = weights.sum()
    if float(denominator.detach()) <= 0:
        return logits.sum()*0
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none")*weights).sum()/denominator
    probability = torch.sigmoid(logits)
    intersection = (probability*target*weights).sum()
    dice = 1-(2*intersection+1)/((probability*weights).sum()+(target*weights).sum()+1)
    return bce+.5*dice


class PatchDataset(Dataset):
    def __init__(self, scenes, normalization, *, patch_size=128, patches_per_scene=16, seed=20261028):
        if patch_size < 32 or patch_size % STRIDE or patches_per_scene < 1:
            raise ValueError("Use patch_size >=32 divisible by8 and positive patch budget")
        self.scenes, self.normalization, self.patch_size = scenes, normalization, patch_size
        groups = defaultdict(list)
        self.anchors = {}
        for i, scene in enumerate(scenes):
            known = scene["valid"] & (scene["labels"] != 255) & (scene["weights"] > 0)
            if known.any():
                groups[scene["group"]].append(i)
                self.anchors[i] = [np.argwhere(known & (scene["labels"] == code)) for code in (0, 1)]
        if not groups:
            raise ValueError("No supervised training patches")
        rng = np.random.default_rng(seed)
        keys = sorted(groups)
        self.samples = []
        for j in range(patches_per_scene*len(scenes)):
            group = keys[j % len(keys)]
            i = int(rng.choice(groups[group]))
            possible = [a for a in self.anchors[i] if len(a)]
            anchor_pool = possible[int(rng.integers(len(possible)))]
            row, col = anchor_pool[int(rng.integers(len(anchor_pool)))]
            self.samples.append((i, int(row), int(col), int(rng.integers(4)), bool(rng.integers(2)), bool(rng.integers(2))))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        i, row, col, rotations, flip_y, flip_x = self.samples[index]
        scene, side = self.scenes[i], self.patch_size
        r0, c0 = row-side//2, col-side//2
        r1, c1 = r0+side, c0+side
        sr0, sc0 = max(r0, 0), max(c0, 0)
        sr1, sc1 = min(r1, scene["valid"].shape[0]), min(c1, scene["valid"].shape[1])
        dest = (slice(sr0-r0, sr1-r0), slice(sc0-c0, sc1-c0))
        x = np.zeros((2, side, side), np.float32)
        target, weights = np.zeros((side, side), np.float32), np.zeros((side, side), np.float32)
        valid = scene["valid"][sr0:sr1, sc0:sc1]
        labels = scene["labels"][sr0:sr1, sc0:sc1]
        x[:, dest[0], dest[1]] = normalize_power(scene["power"][:, sr0:sr1, sc0:sc1], valid, self.normalization)
        target[dest] = labels == 1
        weights[dest] = scene["weights"][sr0:sr1, sc0:sc1]*valid*(labels != 255)
        def augment(array):
            array = np.rot90(array, rotations, axes=(-2, -1))
            if flip_y: array = np.flip(array, axis=-2)
            if flip_x: array = np.flip(array, axis=-1)
            return torch.from_numpy(np.ascontiguousarray(array))
        return augment(x), augment(target[None]), augment(weights[None])


def _confusion(truth, score, weights, threshold):
    pred, truth = score >= threshold, truth == 1
    tp, fp = float(weights[truth & pred].sum()), float(weights[~truth & pred].sum())
    fn, tn = float(weights[truth & ~pred].sum()), float(weights[~truth & ~pred].sum())
    return {"tp_weight": tp, "fp_weight": fp, "fn_weight": fn, "tn_weight": tn,
            "precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "iou": tp/(tp+fp+fn) if tp+fp+fn else None,
            "false_positive_rate": fp/(fp+tn) if fp+tn else None}


def validation_scores(model, scenes, normalization, device, *, tile_size=256):
    records, scores, losses = [], {}, defaultdict(list)
    for scene in scenes:
        score = _predict_scores(model, scene["power"], scene["valid"], normalization, device, tile_size=tile_size)
        known = scene["valid"] & classification_support(scene["valid"]) & (scene["labels"] != 255) & (scene["weights"] > 0)
        scores[scene["id"]] = score
        if not known.any():
            records.append({"id": scene["id"], "group": scene["group"], "region": scene["region"], "state": "no_retained_reference", "loss": None})
            continue
        p = np.clip(score[known].astype(np.float64), 1e-6, 1-1e-6)
        y, w = (scene["labels"][known] == 1).astype(np.float64), scene["weights"][known].astype(np.float64)
        bce = -float((w*(y*np.log(p)+(1-y)*np.log(1-p))).sum()/w.sum())
        dice = 1-float((2*(p*y*w).sum()+1)/((p*w).sum()+(y*w).sum()+1))
        loss = bce+.5*dice
        losses[scene["region"]].append(loss)
        records.append({"id": scene["id"], "group": scene["group"], "region": scene["region"], "state": "scored", "loss": loss,
                        "known_cells": int(known.sum()), "water_cells": int(y.sum()), "metrics_at_0_5": _confusion(y, p, w, .5)})
    if not losses:
        raise ValueError("Validation has no known feature-supported cells")
    return float(np.mean([np.mean(values) for values in losses.values()])), records, scores


def threshold_reports(scenes, scores, thresholds):
    reports = []
    for threshold in thresholds:
        scene_reports, regions, pooled = [], defaultdict(list), []
        for scene in scenes:
            known = scene["valid"] & classification_support(scene["valid"]) & (scene["labels"] != 255) & (scene["weights"] > 0)
            if not known.any():
                continue
            y, s, w = scene["labels"][known], scores[scene["id"]][known], scene["weights"][known].astype(np.float64)
            metric = _confusion(y, s, w, threshold)
            scene_reports.append({"id": scene["id"], "region": scene["region"], **metric})
            regions[scene["region"]].append((y, s, w))
            pooled.append((y, s, w))
        def aggregate(items):
            return _confusion(*[np.concatenate([item[i] for item in items]) for i in range(3)], threshold)
        reports.append({"threshold": float(threshold), "overall": aggregate(pooled), "scenes": scene_reports,
                        "regions": {region: aggregate(items) for region, items in regions.items()}})
    return reports


def _save_weights(path, model):
    torch.save({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, path)


def train_unet(manifest_path, output_dir, *, epochs=20, patch_size=128, patches_per_scene=16,
               batch_size=8, learning_rate=1e-3, base_channels=24, device="auto", seed=20261028,
               threshold_grid=(.1, .2, .3, .4, .5, .6, .7, .8, .9), allow_synthetic_fixture=False):
    """Fit one fixed model; select epoch on macro-region validation loss, never test.

    The returned checkpoint defaults to threshold0.5. The integrator chooses the
    family/operating threshold using the saved validation-only threshold reports.
    """
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Preserve prior training evidence; choose a new output directory")
    if not 1 <= epochs <= 200 or batch_size < 1 or patches_per_scene < 1:
        raise ValueError("Invalid fixed training budget")
    device = choose_device(device)
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    manifest, split = development_scenes(manifest_path, allow_synthetic_fixture=allow_synthetic_fixture)
    output_dir.mkdir(parents=True)
    normalization = fit_normalization(split["train"], seed=seed)
    plan = {"created_at": utcnow(), "manifest_sha256": sha256(manifest_path), "epochs": epochs,
            "patch_size": patch_size, "patches_per_scene": patches_per_scene, "batch_size": batch_size,
            "base_channels": base_channels, "learning_rate": learning_rate, "seed": seed,
            "normalization": normalization, "device": str(device), "threshold_grid": list(threshold_grid),
            "epoch_selection": "minimum mean region validation BCE+0.5Dice; no test access",
            "sampling": "equal geographic group patch budget; random scene per group; balanced labelled class anchors when both exist",
            "augmentation": "horizontal/vertical flips and multiples of90degree only",
            "train_scene_ids": [s["id"] for s in split["train"]],
            "validation_scene_ids": [s["id"] for s in split["validation"]],
            "scenes_without_supervised_training_support": [s["id"] for s in split["train"] if not
                (s["valid"] & (s["labels"] != 255) & (s["weights"] > 0)).any()],
            "test_paths_opened": False, "implementation_sha256": sha256(__file__)}
    write_json(output_dir/"plan.json", plan)
    model = SmallUNet(base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best, best_epoch = [], float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        dataset = PatchDataset(split["train"], normalization, patch_size=patch_size,
                               patches_per_scene=patches_per_scene, seed=seed+epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
                            pin_memory=device.type == "cuda")
        model.train(); running, batches = 0., 0
        for x, target, weights in loader:
            x, target, weights = x.to(device), target.to(device), weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = masked_bce_dice(model(x), target, weights)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            scaler.step(optimizer); scaler.update()
            running += float(loss.detach()); batches += 1
        val_loss, records, _ = validation_scores(model, split["validation"], normalization, device)
        row = {"epoch": epoch, "train_loss": running/max(batches, 1), "validation_macro_region_loss": val_loss,
               "validation_scenes": records, "elapsed_seconds": time.perf_counter()-started}
        history.append(row)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            _save_weights(output_dir/"weights.pt", model)
        write_json(output_dir/"history.json", {"epochs": history, "best_epoch": best_epoch})
        print(f"U-Net epoch {epoch}/{epochs}: train={row['train_loss']:.4f} validation={val_loss:.4f} best={best_epoch}", flush=True)
    state = torch.load(output_dir/"weights.pt", map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    _, records, scores = validation_scores(model, split["validation"], normalization, device)
    reports = threshold_reports(split["validation"], scores, threshold_grid)
    validation_dir = output_dir/"validation-scores"; validation_dir.mkdir()
    for scene in split["validation"]:
        np.savez_compressed(validation_dir/(str(scene["id"]).replace("/", "-").replace("\\", "-")+".npz"),
                            score=scores[scene["id"]])
    metadata = {"schema_version": SCHEMA, "model_type": "unet", "model_id": "nisar-water-unet-round3-v1",
                "feature_version": FEATURE_VERSION, "channel_order": CHANNELS, "units": "gamma0_power",
                "analysis_spacing_m": 20, "base_channels": base_channels,
                "parameter_count": sum(p.numel() for p in model.parameters()), "threshold": .5,
                "decision_policy": {"kind": "threshold", "threshold": .5}, "normalization": normalization,
                "weights_sha256": sha256(output_dir/"weights.pt"), "created_at": utcnow(),
                "training_origin": manifest["data_origin"], "label_kind": manifest["label_kind"],
                "manifest_sha256": sha256(manifest_path), "best_epoch": best_epoch,
                "evaluation_status": "synthetic_fixture" if manifest["data_origin"] == "synthetic_fixture" else "experimental_unet_weak_validation",
                "validation": {"macro_region_loss": best, "scenes": records},
                "test_evaluated": False, "normalization_fitted_on": "training_scenes_only",
                "support_policy": "source validity, finite positive HH/HV, >=80% support in3/7cell windows; exclusions withheld",
                "inference": {"core_default": 256, "halo_pixels": INFERENCE_HALO, "alignment_stride": STRIDE,
                              "edge_policy": "normalized mean padding, fixed BatchNorm evaluation statistics; discard receptive halos"},
                "software": {"torch": torch.__version__, "numpy": np.__version__},
                "implementation_sha256": sha256(__file__)}
    write_json(output_dir/"model.json", metadata)
    write_json(output_dir/"validation-thresholds.json", {"reports": reports, "test_used": False})
    return metadata


class NeuralWaterModel:
    """Existing pipeline-compatible model object using a tensor-only state dict."""
    def __init__(self, directory, *, device="auto"):
        self.directory = Path(directory)
        self.metadata = read_json(self.directory/"model.json")
        if self.metadata.get("schema_version") != SCHEMA or self.metadata.get("model_type") != "unet":
            raise ValueError("Unsupported neural checkpoint schema")
        if self.metadata.get("feature_version") != FEATURE_VERSION or self.metadata.get("channel_order") != CHANNELS:
            raise ValueError("Neural checkpoint physical input contract differs")
        path = self.directory/"weights.pt"
        if path.stat().st_size > 64*1024**2 or sha256(path) != self.metadata["weights_sha256"]:
            raise ValueError("Neural checkpoint size/hash mismatch")
        self.device = choose_device(device)
        self.model = SmallUNet(int(self.metadata["base_channels"])).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=True)
        self.model.eval()
        self.model_id = self.metadata["model_id"]
        self.threshold = float(self.metadata["threshold"])
        self.fingerprint = sha256(self.directory/"model.json")

    def predict(self, observation, *, threshold=None, tile_size=256):
        threshold = self.threshold if threshold is None else float(threshold)
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Threshold must be finite and strictly between zero and one")
        power, valid, _, _ = _validate_arrays(observation.power, np.asarray(observation.valid, bool))
        score = _predict_scores(self.model, power, valid, self.metadata["normalization"], self.device, tile_size=tile_size)
        observed = valid & (observation.weights > 0)
        supported = classification_support(valid)
        classified = observed & supported & ~observation.exclusions
        labels = np.full(valid.shape, 255, np.uint8)
        labels[observed] = 254
        labels[classified] = (score[classified] >= threshold).astype(np.uint8)
        score[~classified] = np.nan
        return labels, score, {"threshold": threshold, "threshold_override": threshold != self.threshold,
                             "decision_policy": {"kind": "threshold", "threshold": threshold},
                             "evaluation_status": self.metadata["evaluation_status"],
                             "score_semantics": "uncalibrated neural model score, not measured probability of water"}

