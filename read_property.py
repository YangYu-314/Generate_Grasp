#!/usr/bin/env python3
"""
Stitches OBJs with correct scaling, maps Lidar points to parts via KD-Tree,
and appends physics properties (density, friction, variance).

Output: (N, 6) numpy array -> [x, y, z, density, friction, variance]
"""

import argparse
import json
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import itertools
from scipy.spatial import cKDTree
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ================= Configuration =================
# Adjust these paths to match your cluster environment
DATASET_ROOT = Path("/cluster/scratch/yangyu1/datasets")
JSON_PATH = DATASET_ROOT / "json"
MESH_PATH = DATASET_ROOT / "mesh"
MATERIAL_PROPS_PATH = Path("/cluster/home/yangyu1/Isaac/Generate_Grasp/material_properties.json")
# Input lidar file name (assumed to be inside some directory, logic handles path)
# Output directory

SCALE_RELAXATION = 0.95
SCALE_ALIGNMENT_SPREAD = 1.2

_MATERIAL_PROPS = None

# ================= Helper Functions (Copied from your script) =================

def load_material_properties() -> Dict[str, Dict[str, float]]:
    global _MATERIAL_PROPS
    if _MATERIAL_PROPS is None:
        try:
            with MATERIAL_PROPS_PATH.open("r") as fh:
                _MATERIAL_PROPS = json.load(fh)
        except FileNotFoundError:
            _MATERIAL_PROPS = {}
    return _MATERIAL_PROPS

def get_material_friction(material_name: str, default_friction: float = 0.5, default_sigma: float = 0.0) -> Tuple[float, float]:
    props = load_material_properties()
    entry = props.get(material_name, {})
    friction = entry.get("friction")
    sigma = entry.get("sigma")
    if friction is None:
        return default_friction, default_sigma
    try:
        return float(friction), float(sigma)
    except Exception:
        return default_friction, default_sigma

def parse_density(value: object, default_density: float = 1000.0) -> float:
    if value is None: return default_density
    if isinstance(value, (int, float)): return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        num = None
        for token in text.replace("/", " ").replace("*", " ").split():
            try:
                num = float(token)
                break
            except ValueError:
                continue
        if num is None: return default_density
        if "g" in text and "cm" in text: return num * 1000.0
        if "kg" in text and "m" in text: return num
        return num
    return default_density

def parse_dimension(value: str) -> np.ndarray:
    if not value: return np.ones(3, dtype=float)
    parts = [chunk.strip() for chunk in value.lower().split("*") if chunk.strip()]
    if len(parts) != 3: return np.ones(3, dtype=float)
    dim_cm = np.array([float(p) for p in parts], dtype=float)
    return dim_cm / 100.0

def read_obj_bounds(path: Path) -> Tuple[np.ndarray, np.ndarray] | None:
    # Optimized numpy reader for bounds
    try:
        mesh = trimesh.load(path, force='mesh', process=False, skip_materials=True)
        return mesh.bounds[0], mesh.bounds[1]
    except Exception:
        return None

def aggregate_bounds(object_id: str, part_labels: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    global_min = np.array([np.inf, np.inf, np.inf], dtype=float)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    found = False
    part_dir = MESH_PATH / object_id / "objs"
    for label in part_labels:
        obj_path = part_dir / f"{label}.obj"
        if not obj_path.exists(): continue
        bounds = read_obj_bounds(obj_path)
        if bounds is None: continue
        global_min = np.minimum(global_min, bounds[0])
        global_max = np.maximum(global_max, bounds[1])
        found = True
    if not found:
        return np.zeros(3, dtype=float), np.ones(3, dtype=float)
    return global_min, global_max

def align_dimension_to_size(dimension: np.ndarray, size: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    dim = np.asarray(dimension, dtype=float)
    size = np.asarray(size, dtype=float)
    if dim.shape != (3,) or size.shape != (3,) or np.any(size <= 1e-9):
        return dim, (0, 1, 2)
    raw_scale = np.where(size > 1e-9, dim / size, 1.0)
    spread = raw_scale.max() / max(raw_scale.min(), 1e-9)
    if spread <= SCALE_ALIGNMENT_SPREAD:
        return dim, (0, 1, 2)
    best_perm = (0, 1, 2)
    best_score = float("inf")
    for perm in itertools.permutations(range(3)):
        perm_dim = dim[list(perm)]
        perm_scale = np.where(size > 1e-9, perm_dim / size, 1.0)
        log_scale = np.log(np.clip(perm_scale, 1e-9, 1e9))
        score = np.ptp(log_scale)
        if score < best_score:
            best_score = score
            best_perm = perm
    return dim[list(best_perm)], best_perm

def safe_scale(dimension: np.ndarray, size: np.ndarray) -> np.ndarray:
    scale = np.ones(3, dtype=float)
    for axis in range(3):
        if size[axis] <= 1e-6 or dimension[axis] <= 0:
            scale[axis] = 1.0
        else:
            raw = dimension[axis] / size[axis]
            scale[axis] = max(raw * SCALE_RELAXATION, 1e-3)
    return scale

def read_object_json(file_name: str) -> Dict[str, object]:
    file_path = JSON_PATH / file_name
    with file_path.open("r") as fh:
        data = json.load(fh)
    return data # Direct return for simpler usage

# ================= Core Processing Class =================

class GeometryPhysicsMapper:
    def __init__(self, object_id: str):
        self.object_id = object_id
        self.tree = None
        self.part_indices = None
        # lookup tables: index -> value
        self.density_lookup = []
        self.friction_lookup = []
        self.friction_var_lookup = []
        self.max_neighbor_dist = np.inf
        
        self._build_index()

    def _build_index(self):
        # 1. Load Metadata
        try:
            meta = read_object_json(f"{self.object_id}.json")
        except FileNotFoundError:
            print(f"[Error] Metadata not found for {self.object_id}")
            return

        parts_meta = meta.get("parts", [])
        part_labels = [p.get("label") for p in parts_meta if "label" in p]
        
        # 2. Compute Scale (Crucial: Must match Isaac Sim logic)
        bbox_min, bbox_max = aggregate_bounds(self.object_id, part_labels)
        size = bbox_max - bbox_min
        dim_raw = parse_dimension(meta.get("dimension", ""))
        dim, _ = align_dimension_to_size(dim_raw, size)
        scale = safe_scale(dim, size)
        diag = np.linalg.norm(size * scale)
        self.max_neighbor_dist = 0.5 * diag + 0.01  # half diagonal + small slack

        rot_x_90 = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.0, -1.0],
                             [0.0, 1.0, 0.0]], dtype=float)
        
        # 3. Load Meshes, Apply Scale, Store Props
        all_points = []
        all_indices = []
        
        part_dir = MESH_PATH / self.object_id / "objs"
        
        for i, p_meta in enumerate(parts_meta):
            label = p_meta.get("label")
            if label is None: continue
            
            obj_path = part_dir / f"{label}.obj"
            if not obj_path.exists(): continue
            
            # Load mesh
            mesh = trimesh.load(obj_path, force='mesh', process=False, skip_materials=True)
            # APPLY SCALE (Crucial step!)
            mesh.apply_scale(scale)
            # Rotate to match USD frame (+90deg about X)
            points = np.asarray(mesh.vertices) @ rot_x_90.T
            
            # Parse Physics
            mat_name = p_meta.get("material", "Plastic")
            print(f"Part {label}: Material={mat_name}")
            friction, sigma = get_material_friction(mat_name)
            density = parse_density(p_meta.get("density"))
            print(f"  -> Density={density:.3g} kg/m^3, Friction={friction:.3g}, Variance={sigma:.3g}")
            
            # Append to lookups
            # Note: 'i' is the index in our lookup tables
            self.friction_lookup.append(friction)
            self.friction_var_lookup.append(sigma)
            self.density_lookup.append(density)
            
            all_points.append(points)
            all_indices.append(np.full(len(points), i, dtype=np.int32))
            
        # 4. Build KD-Tree
        if all_points:
            cloud = np.vstack(all_points)
            self.part_indices = np.concatenate(all_indices)
            # Build tree
            self.tree = cKDTree(cloud)
        else:
            print(f"[Warn] No geometry loaded for {self.object_id}")

    def process_lidar(self, lidar_path: Path) -> np.ndarray:
        """
        Reads lidar npy, queries tree, returns (N, 5) array.
        """
        if self.tree is None:
            return None
        
        # Load Points (N, 3)
        try:
            points = np.load(lidar_path)
            if points.ndim != 2 or points.shape[1] != 3:
                # Handle cases where lidar might be (N, 6) or transposed
                points = points[:, :3]
        except Exception as e:
            print(f"[Error] Failed to load lidar {lidar_path}: {e}")
            return None
            
        if len(points) == 0:
            return np.zeros((0, 5))

        # Query Tree (Batch)
        # k=1, jobs=-1 (use all cores)
        # distance_upper_bound: points too far from mesh get default
        dists, hits = self.tree.query(points, k=1, workers=1, distance_upper_bound=self.max_neighbor_dist)
         
        # Prepare result array: [x, y, z, density, friction, variance]
        result = np.zeros((len(points), 6), dtype=np.float32)
        result[:, :3] = points
        
        # Handle hits
        valid_mask = np.isfinite(dists) & (hits < len(self.part_indices))
        
        if np.any(valid_mask):
            # Map tree index -> part list index
            valid_hits = hits[valid_mask]
            part_idxs = self.part_indices[valid_hits]
            
            # Map part list index -> physics properties
            densities = np.array(self.density_lookup)[part_idxs]
            frictions = np.array(self.friction_lookup)[part_idxs]
            variance = np.array(self.friction_var_lookup)[part_idxs]
            
            result[valid_mask, 3] = densities
            result[valid_mask, 4] = frictions
            result[valid_mask, 5] = variance
            
        # Handle outliers (optional: set to 0 or mean)
        # Current logic leaves them as 0.0
        
        return result


def _save_class_plot(points: np.ndarray, values: np.ndarray, title: str, out_path: Path) -> None:
    """Save a 3D scatter colored by discrete classes (unique values mapped to a palette)."""
    if points.size == 0 or values.size == 0:
        return

    uniq = sorted(np.unique(values))
    if not uniq:
        return

    index_map = {v: i for i, v in enumerate(uniq)}
    mapped = np.array([index_map.get(v, -1) for v in values], dtype=np.int32)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=mapped, cmap="tab10", s=1)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(-0.1, 0.1)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.07)
    cbar.set_ticks(np.arange(len(uniq)))
    cbar.set_ticklabels([f"{u:.3g}" for u in uniq])
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# ================= Main Execution =================

def process_object(object_id: str, lidar_file: Path, output_dir: Path):
    mapper = GeometryPhysicsMapper(object_id)
    
    labeled_data = mapper.process_lidar(lidar_file)
    
    if labeled_data is not None:
        out_path = output_dir / f"{object_id}_physical.npy"
        np.save(out_path, labeled_data)
        print(f"[OK] Saved {out_path} shape={labeled_data.shape}")
        pts = labeled_data[:, :3]
        dens = labeled_data[:, 3]
        fric = labeled_data[:, 4]
        var = labeled_data[:, 5]
        _save_class_plot(pts, fric, "Friction classes", output_dir / f"{object_id}_friction.png")
        _save_class_plot(pts, dens, "Density classes", output_dir / f"{object_id}_density.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", help="Specific IDs to process (defaults to auto-discovery in lidar_dir)")
    parser.add_argument("--lidar_dir", type=str, required=True, help="Root directory containing per-id folders with <id>.npy")
    parser.add_argument("--output_root", type=str, default=None, help="Optional output root (defaults to the per-id folder inside lidar_dir)")
    args = parser.parse_args()

    print("Starting processing...")

    lidar_dir = Path(args.lidar_dir)

    # Identify objects to process
    if args.ids:
        ids = args.ids
    else:
        ids = []
        for child in lidar_dir.iterdir():
            if not child.is_dir():
                continue
            obj_id = child.name
            if (child / f"{obj_id}.npy").exists():
                ids.append(obj_id)
        ids = sorted(ids)

    for object_id in ids:
        lidar_file = lidar_dir / object_id / f"{object_id}.npy"

        print("Looking for lidar file:", lidar_file)

        if lidar_file.exists():
            out_dir = Path(args.output_root) / object_id if args.output_root else lidar_file.parent
            out_path = out_dir / f"{object_id}_physical.npy"
            if out_path.exists():
                print(f"[Skip] Physical output already exists for {object_id}: {out_path}")
                continue
            print(f"Processing {object_id}...")
            out_dir.mkdir(parents=True, exist_ok=True)
            process_object(object_id, lidar_file, out_dir)
        else:
            print(f"[Skip] Lidar file not found for {object_id}")

if __name__ == "__main__":
    main()
