from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import yaml

DATASET_ROOT = Path(__file__).resolve().parent
JSON_PATH = DATASET_ROOT / "json"
MESH_PATH = DATASET_ROOT / "mesh"
OUTPUT_PATH = DATASET_ROOT.parent / "Generator" / "configs"/ "scene"
CACHE_DIR = DATASET_ROOT / "cache"
CACHE_PATH = CACHE_DIR / "bounds.json"
SCALE_RELAXATION = 0.95  # shrink slightly to avoid too tight fitting
RARE_MATERIAL_THRESHOLD = 50
MIN_MATERIAL_TYPES_PER_SCENE = 3


default_rng = np.random.default_rng


@dataclass
class ObjectInfo:
    file_id: str
    object_name: str
    dimension: np.ndarray
    materials: List[str]
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    size: np.ndarray
    scale: np.ndarray

    def to_debug_dict(self) -> Dict[str, object]:
        return {
            "file_id": self.file_id,
            "name": self.object_name,
            "size": self.size.round(4).tolist(),
            "scale": self.scale.round(4).tolist(),
            "materials": self.materials,
        }


@dataclass
class SceneDefinition:
    name: str
    split: str
    objects: List[ObjectInfo]

    def to_yaml_dict(self) -> Dict[str, object]:
        positions, rotations = prepare_positions_and_rotations(self.objects)
        debug_materials = sorted({m for obj in self.objects for m in obj.materials})
        data = {
            "Name": f"Auto Scene {self.name}",
            "Description": f"Auto-generated {self.split} scene covering {len(debug_materials)} materials.",
            "mode": "table",
            "object_config": {
                "mode": "selected",
                "number": len(self.objects),
                "objects": [],
            },
            "debug": {
                "split": self.split,
                "scene_id": self.name,
                "object_count": len(self.objects),
                "object_ids": [obj.file_id for obj in self.objects],
                "object_names": [obj.object_name for obj in self.objects],
                "materials": debug_materials,
                "material_count": len(debug_materials),
            },
        }

        for idx, obj in enumerate(self.objects):
            entry = {
                "name": obj.file_id,
                "position": positions[idx],
                "rotation": rotations[idx],
                "scale": np.round(obj.scale, 6).tolist(),
                "physics": True,
            }
            data["object_config"]["objects"].append(entry)

        return data


def parse_dimension(value: str) -> np.ndarray:
    if not value:
        print("[Warning] Empty dimension string, defaulting to ones.")
        return np.ones(3, dtype=float)
    parts = [chunk.strip() for chunk in value.lower().split("*") if chunk.strip()]
    if len(parts) != 3:
        print(f"[Warning] Invalid dimension format '{value}', defaulting to ones.")
        return np.ones(3, dtype=float)
    return np.array([float(p) for p in parts], dtype=float)


def read_obj_bounds(path: Path) -> Tuple[np.ndarray, np.ndarray] | None:
    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    has_vertex = False
    try:
        with path.open("r") as fh:
            for line in fh:
                if not line.startswith("v "):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
                mins = np.minimum(mins, xyz)
                maxs = np.maximum(maxs, xyz)
                has_vertex = True
    except FileNotFoundError:
        return None

    if not has_vertex:
        print(f"[Warning] No vertices found in OBJ file: {path}")
        return None
    return mins, maxs


def aggregate_bounds(object_id: str, part_labels: Iterable[int], cache: Dict[str, Dict[str, List[float]]]) -> Tuple[np.ndarray, np.ndarray]:
    if object_id in cache:
        cached = cache[object_id]
        return np.array(cached["min"], dtype=float), np.array(cached["max"], dtype=float)

    global_min = np.array([np.inf, np.inf, np.inf], dtype=float)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    part_dir = MESH_PATH / object_id / "objs"
    found = False

    for label in part_labels:
        obj_path = part_dir / f"{label}.obj"
        bounds = read_obj_bounds(obj_path)
        if bounds is None:
            print(f"[Warning] Missing or invalid OBJ file: {obj_path}")
            continue
        mins, maxs = bounds
        global_min = np.minimum(global_min, mins)
        global_max = np.maximum(global_max, maxs)
        found = True

    if not found:
        global_min = np.zeros(3, dtype=float)
        global_max = np.ones(3, dtype=float)

    cache[object_id] = {"min": global_min.tolist(), "max": global_max.tolist()}
    return global_min, global_max


def safe_scale(dimension: np.ndarray, size: np.ndarray) -> np.ndarray:
    scale = np.ones(3, dtype=float)
    for axis in range(3):
        if size[axis] <= 1e-6:
            scale[axis] = 1.0
        elif dimension[axis] <= 0:
            scale[axis] = 1.0
        else:
            raw = dimension[axis] / size[axis]
            scale[axis] = max(raw * SCALE_RELAXATION, 1e-3)
    return scale


def load_bounds_cache() -> Dict[str, Dict[str, List[float]]]:
    if CACHE_PATH.exists():
        with CACHE_PATH.open("r") as fh:
            return json.load(fh)
    return {}


def flush_bounds_cache(cache: Dict[str, Dict[str, List[float]]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w") as fh:
        json.dump(cache, fh, indent=2)


def read_object_json(file_name: str) -> Dict[str, object]:
    file_path = JSON_PATH / file_name
    with file_path.open("r") as fh:
        data = json.load(fh)

    info = {
        "file_id": file_name.split(".")[0],
        "object_name": data.get("object_name", ""),
        "dimension": data.get("dimension", ""),
        "parts": [],
    }

    for part in data.get("parts", []):
        info["parts"].append(
            {
                "part_id": part.get("label", -1),
                "part_name": part.get("name", ""),
                "part_material": part.get("material", "Unknown"),
            }
        )
    return info


def extract_unique_materials(meta: Dict[str, object]) -> List[str]:
    materials = {
        part.get("part_material")
        for part in meta.get("parts", [])
        if part.get("part_material")
    }
    result = sorted(materials)
    if not result:
        result = ["Unknown"]
    return result


def build_object_infos() -> List[ObjectInfo]:
    cache = load_bounds_cache()
    object_infos: List[ObjectInfo] = []

    for json_file in sorted(JSON_PATH.glob("*.json")):
        meta = read_object_json(json_file.name)
        parts = meta.get("parts", [])
        labels = [part["part_id"] for part in parts if part["part_id"] != -1]
        bbox_min, bbox_max = aggregate_bounds(meta["file_id"], labels, cache)
        size = bbox_max - bbox_min
        dimension = parse_dimension(meta.get("dimension", "")) / 100.0
        scale = safe_scale(dimension, size)
        materials = extract_unique_materials(meta)

        object_infos.append(
            ObjectInfo(
                file_id=meta["file_id"],
                object_name=meta.get("object_name", meta["file_id"]),
                dimension=dimension,
                materials=materials,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                size=size,
                scale=scale,
            )
        )

    flush_bounds_cache(cache)
    return object_infos


def split_train_test(objects: Sequence[ObjectInfo], train_ratio: float = 0.8, seed: int = 0) -> Tuple[List[ObjectInfo], List[ObjectInfo]]:
    rng = random.Random(seed)
    shuffled = list(objects)
    rng.shuffle(shuffled)

    material_total = Counter()
    for obj in shuffled:
        material_total.update(obj.materials)

    required_train = {
        material: max(1, math.ceil(count * train_ratio))
        for material, count in material_total.items()
    }

    train: List[ObjectInfo] = []
    test: List[ObjectInfo] = []
    material_in_train: Dict[str, int] = defaultdict(int)

    # Prioritize rarer materials first
    def rarity_score(obj: ObjectInfo) -> int:
        return min(material_total[m] for m in obj.materials)

    ordered = sorted(shuffled, key=rarity_score)

    def add_to_train(obj: ObjectInfo) -> None:
        train.append(obj)
        for m in obj.materials:
            material_in_train[m] += 1

    for obj in ordered:
        needs = [m for m in obj.materials if material_in_train[m] < required_train[m]]
        if needs:
            add_to_train(obj)
        else:
            test.append(obj)

    target_train = max(1, int(len(objects) * train_ratio))
    while len(train) < target_train and test:
        add_to_train(test.pop())

    # bring back any test objects that introduce unseen materials
    remaining_test: List[ObjectInfo] = []
    for obj in test:
        if any(material_in_train[m] == 0 for m in obj.materials):
            add_to_train(obj)
        else:
            remaining_test.append(obj)
    test = remaining_test

    return train, test


class SceneGenerator:
    def __init__(
        self,
        objects_per_scene_range: Tuple[int, int] = (10, 10),
        rare_materials: Optional[Set[str]] = None,
        rare_objects: Optional[Sequence[ObjectInfo]] = None,
        rare_duplicate_limit: int = 2,
        seed: int = 0,
    ) -> None:
        lo, hi = objects_per_scene_range
        self.min_objects = max(1, min(lo, hi))
        self.max_objects = max(self.min_objects, max(lo, hi))
        self.rare_materials = rare_materials or set()
        self.rare_objects = list(rare_objects) if rare_objects else []
        self.rare_duplicate_limit = max(0, rare_duplicate_limit)
        self.rare_reuse_counts: Dict[str, int] = defaultdict(int)
        self.rng = random.Random(seed)

    def build_scenes(self, objects: Sequence[ObjectInfo], split: str) -> List[SceneDefinition]:
        remaining = list(objects)
        scenes: List[SceneDefinition] = []
        scene_idx = 0

        while remaining:
            selected = self._select_scene_objects(remaining)
            for obj in selected:
                if obj in remaining:
                    remaining.remove(obj)
            scene_name = f"{split}_{scene_idx:03d}"
            scenes.append(SceneDefinition(name=scene_name, split=split, objects=selected))
            scene_idx += 1

        return scenes

    def _select_scene_objects(self, pool: List[ObjectInfo]) -> List[ObjectInfo]:
        if not pool:
            return []
        picked: List[ObjectInfo] = []
        covered: set[str] = set()
        available = list(pool)
        target_count = self.rng.randint(self.min_objects, self.max_objects)

        while available and len(picked) < target_count:
            best_obj = max(
                available,
                key=lambda obj: (
                    len(set(obj.materials) - covered),
                    len(obj.materials),
                    -len(obj.file_id),
                ),
            )
            picked.append(best_obj)
            covered.update(best_obj.materials)
            available.remove(best_obj)
        if len(covered) < MIN_MATERIAL_TYPES_PER_SCENE:
            self._inject_additional_fallback(picked, covered, target_count)
        return picked

    def _is_rare(self, obj: ObjectInfo) -> bool:
        return bool(set(obj.materials) & self.rare_materials)

    def _inject_additional_fallback(self, picked: List[ObjectInfo], covered: set[str], target_count: int) -> None:
        candidate_pool = self.rare_objects if self.rare_objects else []
        candidate_pool = [obj for obj in candidate_pool if obj not in picked]
        if not candidate_pool:
            return
        self.rng.shuffle(candidate_pool)

        while len(covered) < MIN_MATERIAL_TYPES_PER_SCENE and candidate_pool and len(picked) < target_count:
            obj = candidate_pool.pop()
            if self.rare_reuse_counts[obj.file_id] >= self.rare_duplicate_limit:
                continue
            picked.append(obj)
            covered.update(obj.materials)
            self.rare_reuse_counts[obj.file_id] += 1


def prepare_positions_and_rotations(
    objects: Sequence[ObjectInfo],
    radius: float = 0.15,
    margin: float = 0.02,
) -> Tuple[List[List[float]], List[List[float]]]:
    count = len(objects)
    if count == 0:
        return [], []

    def alignment(size: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        min_idx = int(np.argmin(size))
        if min_idx == 0:  # X is shortest -> bring to Z by -90 deg around Y
            return size[[2, 1, 0]], [0.0, -90.0, 0.0]
        if min_idx == 1:  # Y shortest -> bring to Z by +90 deg around X
            return size[[0, 2, 1]], [90.0, 0.0, 0.0]
        return size.copy(), [0.0, 0.0, 0.0]

    positions: List[List[float]] = []
    rotations: List[List[float]] = []
    top_height = 0.0

    for idx in range(count):
        aligned_size, rotation = alignment(objects[idx].size)
        obj_height = float(np.clip(aligned_size[2] * objects[idx].scale[2], 0.0, np.inf))
        if not (math.isfinite(obj_height) and obj_height > 0):
            obj_height = margin

        drop_height = round(top_height + margin, 4)
        top_height = drop_height + obj_height

        angle = 2 * math.pi * idx / count
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        positions.append([round(x, 4), round(y, 4), drop_height])
        rotations.append(rotation)

    return positions, rotations


def generate_scene_configs(
    objects_per_scene: Tuple[int, int] | int = 10,
    train_ratio: float = 0.8,
    seed: int = 0,
    ) -> None:
    
    object_infos = build_object_infos()
    material_counts = Counter()
    for obj in object_infos:
        material_counts.update(obj.materials)
    rare_materials = {
        material
        for material, count in material_counts.items()
        if count <= RARE_MATERIAL_THRESHOLD
    }

    train_objs, test_objs = split_train_test(object_infos, train_ratio=train_ratio, seed=seed)
    train_rare_objects = [obj for obj in train_objs if set(obj.materials) & rare_materials]
    test_rare_objects = [obj for obj in test_objs if set(obj.materials) & rare_materials]

    print("=== Object Split Summary ===")
    print(f"Total objects: {len(object_infos)}")
    print(f"Train objects: {len(train_objs)}")
    print(f"Test objects: {len(test_objs)}")
    print("============================")

    if isinstance(objects_per_scene, int):
        range_tuple = (objects_per_scene, objects_per_scene)
    else:
        range_tuple = objects_per_scene

    train_generator = SceneGenerator(
        objects_per_scene_range=range_tuple,
        rare_materials=rare_materials,
        rare_objects=train_rare_objects,
        seed=seed,
    )
    test_generator = SceneGenerator(
        objects_per_scene_range=range_tuple,
        rare_materials=rare_materials,
        rare_objects=test_rare_objects,
        seed=seed + 1,
    )
    train_scenes = train_generator.build_scenes(train_objs, split="train")
    test_scenes = test_generator.build_scenes(test_objs, split="test")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    for scene in train_scenes + test_scenes:
        output_file = OUTPUT_PATH / f"{scene.name}.yaml"
        with output_file.open("w") as fh:
            yaml.safe_dump(scene.to_yaml_dict(), fh, sort_keys=False)

    summary = {
        "train_scenes": len(train_scenes),
        "test_scenes": len(test_scenes),
        "train_objects": len(train_objs),
        "test_objects": len(test_objs),
    }

    print("=== Scene Generation Summary ===")
    print(f"Generated train scenes: {len(train_scenes)}")
    print(f"Generated test scenes: {len(test_scenes)}")
    print("================================")


if __name__ == "__main__":
    generate_scene_configs()
