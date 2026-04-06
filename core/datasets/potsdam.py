"""
Potsdam RGBIR + DSM dataset.

Adapted from roy_emsgcn/core/datasets/potsdam.py.
Drops pytorch_lightning dependency — uses plain PyTorch DataLoader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import kornia.augmentation as K_aug
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

IGNORE_INDEX = 0

OFFICIAL_TRAIN_TILES = (
    "top_potsdam_2_10", "top_potsdam_2_11", "top_potsdam_2_12",
    "top_potsdam_3_10", "top_potsdam_3_11", "top_potsdam_3_12",
    "top_potsdam_4_10", "top_potsdam_4_11", "top_potsdam_4_12",
    "top_potsdam_5_10", "top_potsdam_5_11", "top_potsdam_5_12",
    "top_potsdam_6_10", "top_potsdam_6_11", "top_potsdam_6_12",
    "top_potsdam_6_7",  "top_potsdam_6_8",  "top_potsdam_6_9",
    "top_potsdam_7_10", "top_potsdam_7_11", "top_potsdam_7_12",
    "top_potsdam_7_7",  "top_potsdam_7_8",  "top_potsdam_7_9",
)

OFFICIAL_TEST_TILES = (
    "top_potsdam_5_15", "top_potsdam_6_15", "top_potsdam_6_13",
    "top_potsdam_3_13", "top_potsdam_4_14", "top_potsdam_6_14",
    "top_potsdam_5_14", "top_potsdam_2_13", "top_potsdam_4_15",
    "top_potsdam_2_14", "top_potsdam_5_13", "top_potsdam_4_13",
    "top_potsdam_3_14", "top_potsdam_7_13",
)

CLASS_NAMES = (
    "Impervious surfaces",
    "Building",
    "Low vegetation",
    "Tree",
    "Car",
)

RGB_TO_CLASS_ID: dict[tuple[int, int, int], int] = {
    (255, 0, 0): 0,       # Clutter -> ignore (was 1)
    (255, 255, 255): 1,   # Impervious
    (0, 0, 255): 2,       # Building
    (0, 255, 255): 3,     # Low vegetation
    (0, 255, 0): 4,       # Tree
    (255, 255, 0): 5,     # Car
}

RGBIR_DIR = "4_Ortho_RGBIR"
DSM_DIR = "1_DSM"
LABEL_DIR = "5_Labels_all"

_IGNORE_SUFFIXES = (".tfw", ".aux", ".xml", ".ovr")


@dataclass(frozen=True)
class TilePaths:
    tile_id: str
    rgbir_path: Path
    dsm_path: Path
    label_path: Path


@dataclass(frozen=True)
class PatchRecord:
    tile_id: str
    rgbir_path: Path
    dsm_path: Path
    label_path: Path
    window: Window


def _is_raster_candidate(path: Path) -> bool:
    if not path.is_file():
        return False
    return not any(path.name.lower().endswith(sfx) for sfx in _IGNORE_SUFFIXES)


def _pick_tile_file(folder: Path, prefix: str) -> Path | None:
    matches = sorted(
        p for p in folder.glob(f"{prefix}*") if _is_raster_candidate(p)
    )
    return matches[0] if matches else None


def _tile_id_to_dsm_prefix(tile_id: str) -> str:
    parts = tile_id.split("_")
    return f"dsm_potsdam_{int(parts[-2]):02d}_{int(parts[-1]):02d}"


def _rgb_mask_to_class_ids(label: np.ndarray,
                           ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    rgb = np.moveaxis(label[:3], 0, -1).astype(np.uint8)
    out = np.full(rgb.shape[:2], ignore_index, dtype=np.int64)
    for color, class_id in RGB_TO_CLASS_ID.items():
        out[np.all(rgb == color, axis=-1)] = class_id
    return out


def _morphological_ndsm_from_dsm(dsm: np.ndarray,
                                  kernel_size: int) -> np.ndarray:
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = torch.from_numpy(dsm).float().unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    eroded = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)
    opened = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=pad)
    ndsm = torch.clamp(x - opened, min=0.0)
    return ndsm.squeeze(0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PotsdamDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        patch_size: int = 256,
        patch_stride: int = 256,
        nodata_threshold: float = 0.7,
        max_sampler_checks: int = 0,
        ndsm_opening_kernel: int = 17,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        assert split in ("train", "test")
        self.root = Path(root)
        self.split = split
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.nodata_threshold = float(nodata_threshold)
        self.max_sampler_checks = int(max_sampler_checks)
        self.ndsm_opening_kernel = int(ndsm_opening_kernel)
        self.ignore_index = int(ignore_index)
        self._handles: dict[str, rasterio.io.DatasetReader] = {}

        self.files = self._collect_files()
        self.records, self.invalid_patches, self.checked_patches = \
            self._build_patch_records()

    def _collect_files(self) -> list[TilePaths]:
        tile_ids = OFFICIAL_TRAIN_TILES if self.split == "train" \
            else OFFICIAL_TEST_TILES
        rgbir_dir = self.root / RGBIR_DIR
        dsm_dir = self.root / DSM_DIR
        label_dir = self.root / LABEL_DIR

        files: list[TilePaths] = []
        for tile_id in tile_ids:
            rgbir = _pick_tile_file(rgbir_dir, f"{tile_id}_RGBIR")
            dsm = _pick_tile_file(dsm_dir, _tile_id_to_dsm_prefix(tile_id))
            label = _pick_tile_file(label_dir, f"{tile_id}_label")
            if rgbir is None or dsm is None or label is None:
                continue
            files.append(TilePaths(tile_id, rgbir, dsm, label))
        return files

    def _build_patch_records(self):
        records: list[PatchRecord] = []
        invalid = 0
        checked = 0
        max_checks = self.max_sampler_checks if self.max_sampler_checks > 0 \
            else None

        for tile in self.files:
            with rasterio.open(tile.label_path) as src:
                h, w = int(src.height), int(src.width)
                for top in range(0, h - self.patch_size + 1, self.patch_stride):
                    for left in range(0, w - self.patch_size + 1,
                                      self.patch_stride):
                        if max_checks is not None and checked >= max_checks:
                            break
                        window = Window(left, top, self.patch_size,
                                        self.patch_size)
                        lbl = np.asarray(
                            src.read(window=window, masked=True).filled(0))
                        gt = _rgb_mask_to_class_ids(lbl, self.ignore_index)
                        if float((gt == self.ignore_index).mean()) \
                                <= self.nodata_threshold:
                            records.append(PatchRecord(
                                tile.tile_id, tile.rgbir_path,
                                tile.dsm_path, tile.label_path, window))
                        else:
                            invalid += 1
                        checked += 1
                    if max_checks is not None and checked >= max_checks:
                        break
            if max_checks is not None and checked >= max_checks:
                break

        return records, invalid, checked

    def _get_handle(self, path: Path):
        key = str(path)
        if key not in self._handles:
            self._handles[key] = rasterio.open(path)
        return self._handles[key]

    def close(self) -> None:
        for h in self._handles.values():
            h.close()
        self._handles.clear()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        win = rec.window

        ms = np.asarray(
            self._get_handle(rec.rgbir_path).read(window=win, masked=True
                                                   ).filled(0),
            dtype=np.float32)
        dsm = np.asarray(
            self._get_handle(rec.dsm_path).read(1, window=win, masked=True
                                                 ).filled(0),
            dtype=np.float32)
        lbl = np.asarray(
            self._get_handle(rec.label_path).read(window=win, masked=True
                                                   ).filled(0))

        gt = _rgb_mask_to_class_ids(lbl, self.ignore_index)
        ndsm = _morphological_ndsm_from_dsm(dsm, self.ndsm_opening_kernel)

        ignore_mask = gt == self.ignore_index
        ms[:, ignore_mask] = 0.0
        ndsm[:, ignore_mask] = 0.0

        return {
            "ms": torch.from_numpy(ms).float(),
            "ndsm": torch.from_numpy(ndsm).float(),
            "gt": torch.from_numpy(gt).long(),
            "tile_id": rec.tile_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataModule (plain PyTorch, no lightning)
# ─────────────────────────────────────────────────────────────────────────────

class PotsdamDataModule:

    def __init__(self, cfg: dict):
        ds = cfg["dataset"]
        self.root = ds["root"]
        self.batch_size = int(ds["batch_size"])
        self.num_workers = int(ds["num_workers"])
        self.patch_size = int(ds["patch_size"])
        self.patch_stride = int(ds.get("patch_stride", self.patch_size))
        self.nodata_threshold = float(ds.get("nodata_threshold", 0.7))
        self.max_sampler_checks = int(ds.get("max_sampler_checks", 0))
        self.ignore_index = int(ds.get("ignore_index", IGNORE_INDEX))
        self.ndsm_opening_kernel = int(ds.get("ndsm_opening_kernel", 17))
        self.val_fraction = float(ds.get("split", {}).get("val_fraction", 0.2))
        self.split_seed = int(ds.get("split", {}).get("seed", 42))
        self.label_fraction = float(ds.get("label_fraction", 1.0))

        # Band config
        bands = ds.get("bands", {})
        self.wavelengths_nm = bands.get("wavelengths_nm",
                                        [650.0, 560.0, 450.0, 840.0])
        self.rgb_indices = tuple(bands.get("rgb_indices", [0, 1, 2]))

        # Normalization
        norm = ds.get("normalization")
        if norm:
            self.ms_mean = torch.tensor(norm["ms"]["mean"], dtype=torch.float32)
            self.ms_std = torch.tensor(norm["ms"]["std"], dtype=torch.float32)
            self.ndsm_mean = torch.tensor([norm["ndsm"]["mean"]],
                                          dtype=torch.float32)
            self.ndsm_std = torch.tensor([norm["ndsm"]["std"]],
                                         dtype=torch.float32)
        else:
            self.ms_mean = self.ms_std = None
            self.ndsm_mean = self.ndsm_std = None

        # Augmentations
        self.transform = K_aug.AugmentationSequential(
            K_aug.RandomHorizontalFlip(p=0.5),
            K_aug.RandomVerticalFlip(p=0.5),
            K_aug.RandomResizedCrop(
                size=(self.patch_size, self.patch_size),
                scale=(0.8, 1.0), p=0.5),
            data_keys=["image", "image", "mask"],
        )

        self.num_classes = len(ds.get("class_names", CLASS_NAMES))
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        train_full = PotsdamDataset(
            root=self.root, split="train",
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            nodata_threshold=self.nodata_threshold,
            max_sampler_checks=self.max_sampler_checks,
            ndsm_opening_kernel=self.ndsm_opening_kernel,
            ignore_index=self.ignore_index,
        )
        train_len = max(1, int(round(
            (1.0 - self.val_fraction) * len(train_full))))
        val_len = len(train_full) - train_len
        gen = torch.Generator().manual_seed(self.split_seed)
        self.train_dataset, self.val_dataset = random_split(
            train_full, [train_len, val_len], generator=gen)

        # Subsample training set if label_fraction < 1.0
        if self.label_fraction < 1.0:
            subset_size = max(1, int(len(self.train_dataset) * self.label_fraction))
            discard_size = len(self.train_dataset) - subset_size
            gen2 = torch.Generator().manual_seed(self.split_seed)
            self.train_dataset, _ = random_split(
                self.train_dataset, [subset_size, discard_size], generator=gen2)

        self.test_dataset = PotsdamDataset(
            root=self.root, split="test",
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            nodata_threshold=self.nodata_threshold,
            max_sampler_checks=self.max_sampler_checks,
            ndsm_opening_kernel=self.ndsm_opening_kernel,
            ignore_index=self.ignore_index,
        )

        print(f"[Potsdam] Train: {len(self.train_dataset)} | "
              f"Val: {len(self.val_dataset)} | "
              f"Test: {len(self.test_dataset)}")

    @staticmethod
    def _collate(batch):
        ms = torch.stack([s["ms"] for s in batch])
        ndsm = torch.stack([s["ndsm"] for s in batch])
        gt = torch.stack([s["gt"] for s in batch])
        return ms, ndsm, gt

    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=self._collate, pin_memory=True)

    def val_loader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self._collate, pin_memory=True)

    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self._collate, pin_memory=True)

    def normalize(self, ms, ndsm, device=None):
        """Apply normalization in-place on GPU tensors."""
        if self.ms_mean is not None:
            mean = self.ms_mean.to(device or ms.device).view(1, -1, 1, 1)
            std = self.ms_std.to(device or ms.device).view(1, -1, 1, 1)
            ms = (ms - mean) / std.clamp_min(1e-6)
        if self.ndsm_mean is not None:
            mean = self.ndsm_mean.to(device or ndsm.device).view(1, -1, 1, 1)
            std = self.ndsm_std.to(device or ndsm.device).view(1, -1, 1, 1)
            ndsm = (ndsm - mean) / std.clamp_min(1e-6)
        return ms, ndsm

    def augment(self, ms, ndsm, gt):
        """Apply training augmentations (on GPU)."""
        gt = gt.unsqueeze(1).float()
        ms, ndsm, gt = self.transform(ms, ndsm, gt)
        # Kornia may add a leading dim — squeeze back to (B, C, H, W)
        if ms.ndim == 5:
            ms = ms.squeeze(1)
        if ndsm.ndim == 5:
            ndsm = ndsm.squeeze(1)
        if gt.ndim == 5:
            gt = gt.squeeze(1)
        gt = gt.squeeze(1).long()
        return ms, ndsm, gt
