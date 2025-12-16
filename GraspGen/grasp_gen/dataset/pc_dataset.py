# grasp_gen/dataset/lidar_pick_dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh.transformations as tra
from grasp_gen.dataset.dataset_utils import ObjectGraspDataset, get_rotation_augmentation
from grasp_gen.dataset.exceptions import DataLoaderError
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)

InputMode = Literal["xyz", "intensity", "physical"]


def _safe_load_npy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    arr = np.load(str(path))
    if arr.ndim != 2 or arr.shape[0] <= 0:
        raise ValueError(f"Bad npy shape: {path} -> {arr.shape}")
    return arr


def load_object_grasp_datapoint_pc(
    object_id: str,
    grasp_root_dir: str | Path,
    json_filename: Optional[str] = None,
    load_discriminator_dataset: bool = False,
    gripper_info=None,
    min_pos_grasps_gen: int = 5,
    min_neg_grasps_dis: int = 5,
    min_pos_grasps_dis: int = 5,
):
    root = Path(grasp_root_dir)
    jpath = Path(json_filename) if json_filename is not None else root / object_id / f"{object_id}.json"

    if not jpath.is_file():
        return DataLoaderError.GRASPS_FILE_NOT_FOUND, None

    try:
        grasp_json = json.load(open(jpath, "r"))
        object_meta = grasp_json["object"]
        object_scale = object_meta.get("scale", 1.0)

        grasps = grasp_json["grasps"]

        grasp_poses = np.asarray(grasps["transforms"], dtype=np.float32)
        grasp_mask = np.array(grasps["object_in_gripper"], dtype=bool)

        positive_grasps = grasp_poses[grasp_mask]
        negative_grasps = None
        if load_discriminator_dataset:
            not_success = np.logical_not(grasp_mask)
            negative_grasps = grasp_poses[not_success]
    except Exception:
        return DataLoaderError.GRASPS_FILE_LOAD_ERROR, None

    if positive_grasps.shape[0] < min_pos_grasps_gen:
        return DataLoaderError.INSUFFICIENT_GRASPS_FOR_GENERATOR_DATASET, None
    if load_discriminator_dataset:
        if positive_grasps.shape[0] < min_pos_grasps_dis:
            return DataLoaderError.INSUFFICIENT_GRASPS_FOR_DISCRIMINATOR_DATASET, None
        if negative_grasps is None or negative_grasps.shape[0] < min_neg_grasps_dis:
            return DataLoaderError.INSUFFICIENT_GRASPS_FOR_DISCRIMINATOR_DATASET, None

    contacts = None

    if gripper_info is not None and hasattr(gripper_info, "offset_transform"):
        offset_transform = gripper_info.offset_transform
        positive_grasps = np.array([g @ offset_transform for g in positive_grasps])
        if negative_grasps is not None:
            negative_grasps = np.array([g @ offset_transform for g in negative_grasps])

    object_asset_path = object_meta["file"]
    object_scale = object_meta["scale"]

    return DataLoaderError.SUCCESS, ObjectGraspDataset(
        object_mesh=None,
        positive_grasps=positive_grasps,
        contacts=contacts,
        object_asset_path=object_asset_path,
        object_scale=object_scale,
        negative_grasps=negative_grasps,
        positive_grasps_onpolicy=None,
        negative_grasps_onpolicy=None,
    )


@dataclass
class LidarPickDatasetCfg:
    root: str | Path
    split_txt: str | Path  # train.txt / valid.txt
    input_mode: InputMode = "xyz"
    num_points: int = 8096  # 只使用 *_filtered_8096.npy
    min_num_grasps_per_object: int = 64
    min_neg_grasps_per_object: int = 64
    seed: int = 0
    rotation_aug: bool = False
    load_discriminator_dataset: bool = False
    num_grasps_per_object: int = 64
    scene: str = None
    inference: bool = False
    discriminator_ratio: float = 0.5


class LidarPickDataset(Dataset):
    """
    目标：对齐原 ObjectPickDataset 的“每个样本=一个物体点云 + 若干抓取姿态”
    - points: torch.FloatTensor (N, C)   (C取决于input_mode; feat用arr[:, :])
    - grasps: torch.FloatTensor (G, 4, 4)
    - object_id: str
    """

    def __init__(self, cfg: LidarPickDatasetCfg, split=None, scenes=None, inference=False):
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.split_txt = Path(cfg.split_txt)

        if not self.root.is_dir():
            raise FileNotFoundError(f"root not found: {self.root}")
        if not self.split_txt.is_file():
            raise FileNotFoundError(f"split_txt not found: {self.split_txt}")

        self.object_ids = [
            line.strip() for line in self.split_txt.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if len(self.object_ids) == 0:
            raise ValueError(f"empty split file: {self.split_txt}")

        self._base_rng = np.random.default_rng(cfg.seed)

    def __len__(self) -> int:
        return len(self.object_ids)

    def _apply_rotation_aug(
        self,
        points: np.ndarray,
        grasps: np.ndarray,
        rng: np.random.Generator,
        neg_grasps: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Rotate about point cloud mean using get_rotation_augmentation (no translation)."""
        if points.shape[1] < 3:
            raise ValueError("rotation_aug requires at least xyz channels")

        T_rotation = get_rotation_augmentation(stratified_sampling=False)

        center = points[:, :3].mean(axis=0)
        T_world_to_pcmean = tra.translation_matrix(-center)
        T_pcmean_to_world = tra.inverse_matrix(T_world_to_pcmean)
        T_aug = T_pcmean_to_world @ T_rotation @ T_world_to_pcmean

        pc_rot = tra.transform_points(points[:, :3], T_aug)
        points_aug = points.copy()
        points_aug[:, :3] = pc_rot

        grasps_aug = np.array([T_aug @ g for g in grasps])
        neg_aug = None
        if neg_grasps is not None:
            neg_aug = np.array([T_aug @ g for g in neg_grasps])

        return points_aug, grasps_aug, neg_aug

    def _load_points(self, obj_dir: Path) -> np.ndarray:
        mode = self.cfg.input_mode
        if mode == "xyz":
            arr = _safe_load_npy(obj_dir / f"{obj_dir.name}_filtered_8096.npy")
        elif mode == "intensity":
            arr = _safe_load_npy(obj_dir / f"{obj_dir.name}_intensity_filtered_8096.npy")
        elif mode == "physical":
            arr = _safe_load_npy(obj_dir / f"{obj_dir.name}_physical_filtered_8096.npy")
        else:
            raise ValueError(f"Unknown input_mode: {mode}")

        return arr
    
    def _load_grasps(self, object_id: str) -> ObjectGraspDataset:
        err, object_grasp_data = load_object_grasp_datapoint_pc(
            object_id=object_id,
            grasp_root_dir=self.root,
            json_filename=None,
            load_discriminator_dataset=self.cfg.load_discriminator_dataset,
            gripper_info=None,
            min_pos_grasps_gen=self.cfg.min_num_grasps_per_object,
            min_neg_grasps_dis=self.cfg.min_neg_grasps_per_object,
            min_pos_grasps_dis=self.cfg.min_num_grasps_per_object,
        )
        if err != DataLoaderError.SUCCESS or object_grasp_data is None:
            raise ValueError(f"grasp load error {err.description} for {object_id}")
        return object_grasp_data

    def __getitem__(self, idx: int):
        object_id = self.object_ids[idx]
        obj_dir = self.root / object_id

        try:
            arr = self._load_points(obj_dir)
            object_grasp_data = self._load_grasps(object_id)
        except Exception as e:
            return {"invalid": True, "object_id": object_id, "error": str(e)}

        grasps_all = object_grasp_data.positive_grasps
        neg_grasps = object_grasp_data.negative_grasps if self.cfg.load_discriminator_dataset else None

        G = grasps_all.shape[0]
        K = self.cfg.num_grasps_per_object
        NG = neg_grasps.shape[0] if neg_grasps is not None else 0

        rng = np.random.default_rng(self.cfg.seed + int(idx))

        if not self.cfg.load_discriminator_dataset:
            if G >= K:
                sel = rng.choice(G, size=K, replace=False)
            else:
                sel = rng.choice(G, size=K, replace=True)

            grasps = grasps_all[sel]  # (K,4,4)
            grasp_ids = np.zeros((K, 1), dtype=np.int32)
            labels = np.ones((K, 1), dtype=np.float32)
            
        else:
            K_pos = int(round(K * float(self.cfg.discriminator_ratio)))
            K_pos = max(1, min(K - 1, K_pos))

            if G >= K_pos:
                sel_pos = rng.choice(G, size=K_pos, replace=False)
            else:
                sel_pos = rng.choice(G, size=K_pos, replace=True)
            
            if NG >= K - K_pos:
                sel_neg = rng.choice(NG, size=K - K_pos, replace=False)
            else:
                sel_neg = rng.choice(NG, size=K - K_pos, replace=True)

            grasps = np.concatenate([grasps_all[sel_pos], neg_grasps[sel_neg]], axis=0)  # (K,4,4)
            grasp_ids = np.concatenate([np.zeros((K_pos, 1), dtype=np.int32), np.ones((K - K_pos, 1), dtype=np.int32)], axis=0)
            labels = np.concatenate([np.ones((K_pos, 1), dtype=np.float32), np.zeros((K - K_pos, 1), dtype=np.float32)], axis=0)

        if self.cfg.rotation_aug:
            arr, grasps, neg_grasps = self._apply_rotation_aug(arr, grasps, rng, neg_grasps)


        perm = rng.permutation(K)
        grasps = grasps[perm]
        labels = labels[perm]
        grasp_ids = grasp_ids[perm]

        points_t = torch.from_numpy(arr).float()
        grasps_t = torch.from_numpy(grasps).float()
        grasps_id_t = torch.from_numpy(grasp_ids).long()
        labels_t = torch.from_numpy(labels).float()
        grasps_all_t = torch.from_numpy(grasps_all).float()

        output = {
            "points": points_t,
            "grasps": grasps_t,
            # "grasp_ids": grasps_id_t,
            "grasps_highres": grasps_all_t,
            "object_id": object_id,
        }
        if self.cfg.load_discriminator_dataset:
            output["labels"] = labels_t

        return output

    @classmethod
    def from_config(cls, cfg):
        lidar_cfg = LidarPickDatasetCfg(
            root=cfg.root_dir,
            split_txt=cfg.split_txt,
            input_mode=getattr(cfg, "input_mode", "xyz"),
            num_points=cfg.num_points,
            min_num_grasps_per_object=cfg.min_num_grasps_per_object,
            min_neg_grasps_per_object=cfg.min_neg_grasps_per_object,
            seed=getattr(cfg, "random_seed", 0),
            rotation_aug=cfg.rotation_augmentation,
            load_discriminator_dataset=cfg.load_discriminator_dataset,
            num_grasps_per_object=cfg.num_grasps_per_object,
            discriminator_ratio=getattr(cfg, "discriminator_ratio", 0.5),
        )
        return {"cfg": lidar_cfg}
