#!/usr/bin/env python3
"""
Utility to build scaled, merged OBJ meshes and companion USD files for the
GraspDataGen pipeline from the Dataset assets.

- Reads per-object metadata from Dataset/json/<id>.json
- Computes per-axis scale from declared dimensions and part OBJ bounds
- Merges all part OBJ meshes (already positioned) into a single scaled OBJ
- Emits a simple USD with collision + physics material (friction, density)

Outputs are written under Dataset/converted/{obj,usd}/<id>.{obj,usd}
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import trimesh


ROOT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path("/datasets")
_DEFAULT_DATASET_ROOT = DATASET_ROOT
JSON_PATH = DATASET_ROOT / "json"
MESH_PATH = DATASET_ROOT / "mesh"
OUTPUT_OBJ = DATASET_ROOT / "converted" / "obj"
OUTPUT_USD = DATASET_ROOT / "converted" / "usd"
MATERIAL_DIR = DATASET_ROOT / "material"
MATERIAL_PROPS_PATH = ROOT_DIR / "material_properties.json"

SCALE_RELAXATION = 0.95  # shrink slightly to avoid too tight fitting
# If the ratio between the largest/smallest axis scale is below this, treat axes as already aligned
SCALE_ALIGNMENT_SPREAD = 1.2
_SIM_APP = None
_MATERIAL_PROPS: Dict[str, Dict[str, float]] | None = None


def _normalize_path(path_value: Path | None) -> Path | None:
    """Expand user home in incoming paths while tolerating None."""
    if path_value is None:
        return None
    return path_value.expanduser()


def configure_paths(dataset_root: Path | None = None, output_root: Path | None = None) -> None:
    """Update global IO paths using provided overrides or defaults."""
    global DATASET_ROOT, JSON_PATH, MESH_PATH, OUTPUT_OBJ, OUTPUT_USD, MATERIAL_DIR

    root = _normalize_path(dataset_root) or _DEFAULT_DATASET_ROOT
    DATASET_ROOT = root
    JSON_PATH = DATASET_ROOT / "json"
    MESH_PATH = DATASET_ROOT / "mesh"
    MATERIAL_DIR = DATASET_ROOT / "material"

    out_root = _normalize_path(output_root)
    if out_root is None:
        out_root = DATASET_ROOT / "converted"
    OUTPUT_OBJ = out_root / "obj"
    OUTPUT_USD = out_root / "usd"

# Material lookup copied from Generator/utils/loader.py for consistency
ISAAC_MATERIALS_MDL = {
    "Plastic": ("Plastic", "Plastics/Plastic.mdl"),
    "Metal": ("Steel_Carbon", "Metals/Steel_Carbon.mdl"),
    "Ceramic": ("Ceramic_Smooth_Fired", "Stone/Ceramic_Smooth_Fired.mdl"),
    "Stainless Metal": ("Steel_Stainless", "Metals/Steel_Stainless.mdl"),
    "Glass": ("Clear_Glass", "Glass/Clear_Glass.mdl"),
    "Stainless Steel": ("Steel_Stainless", "Metals/Steel_Stainless.mdl"),
    "Steel": ("Steel_Carbon", "Metals/Steel_Carbon.mdl"),
    "Water": ("Water", "Natural/Water.mdl"),
    "Plastic with foam padding": ("Plastic", "Plastics/Plastic.mdl"),
    "Rubber": ("Rubber_Smooth", "Plastics/Rubber_Smooth.mdl"),
    "Fabric": ("Linen_Beige", "Textiles/Linen_Beige.mdl"),
    "Soil": ("Soil_Rocky", "Natural/Soil_Rocky.mdl"),
    "Foam": ("Rubber_Textured", "Plastics/Rubber_Textured.mdl"),
    "Foam and Leather": ("Leather_Brown", "Textiles/Leather_Brown.mdl"),
    "Rubber and Copper": ("Rubber_Smooth", "Plastics/Rubber_Smooth.mdl"),
    "Foam with leather cover": ("Leather_Brown", "Textiles/Leather_Brown.mdl"),
    "Plastic and Foam": ("Plastic", "Plastics/Plastic.mdl"),
    "Plastic/Fabric": ("Linen_Beige", "Textiles/Linen_Beige.mdl"),
    "Foam with leather or fabric cover": ("Leather_Brown", "Textiles/Leather_Brown.mdl"),
    "Silicone": ("Rubber_Smooth", "Plastics/Rubber_Smooth.mdl"),
    "Aluminum": ("Aluminum_Polished", "Metals/Aluminum_Polished.mdl"),
    "Organic": ("Grass_Countryside", "Natural/Grass_Countryside.mdl"),
    "Foam with Leather Cover": ("Leather_Brown", "Textiles/Leather_Brown.mdl"),
    "Foam and Leatherette": ("Leather_Brown", "Textiles/Leather_Brown.mdl"),
    "Leather": ("Leather_Brown", "Textiles/Leather_Brown.mdl"),
    "Organic Material": ("Grass_Countryside", "Natural/Grass_Countryside.mdl"),
    "Water or Soil": ("Soil_Rocky", "Natural/Soil_Rocky.mdl"),
    "Plastic and Metal": ("Plastic", "Plastics/Plastic.mdl"),
}

DATASET_TO_ISAAC = {
    "Plastic": ("plastic", "paint", "none"),
    "Metal": ("steel", "none", "none"),
    "Ceramic": ("ceramic_glass", "clearcoat", "none"),
    "Stainless Metal": ("steel", "clearcoat", "none"),
    "Glass": ("clear_glass", "clearcoat", "single_sided"),
    "Stainless Steel": ("steel", "clearcoat", "none"),
    "Steel": ("steel", "none", "none"),
    "Water": ("water", "none", "single_sided"),
    "Plastic with foam padding": ("plastic", "paint", "none"),
    "Rubber": ("rubber", "none", "none"),
    "Fabric": ("fabric", "none", "none"),
    "Soil": ("dirt", "none", "none"),
    "Foam": ("plastic", "none", "none"),
    "Foam and Leather": ("leather", "paint_clearcoat", "none"),
    "Rubber and Copper": ("rubber", "none", "none"),
    "Foam with leather cover": ("leather", "paint_clearcoat", "none"),
    "Plastic and Foam": ("plastic", "paint", "none"),
    "Plastic/Fabric": ("fabric", "paint", "none"),
    "Foam with leather or fabric cover": ("leather", "paint_clearcoat", "none"),
    "Silicone": ("rubber", "none", "none"),
    "Aluminum": ("aluminum", "clearcoat", "none"),
    "Organic": ("leaf_grass", "none", "none"),
    "Foam with Leather Cover": ("leather", "paint_clearcoat", "none"),
    "Foam and Leatherette": ("leather", "paint_clearcoat", "none"),
    "Leather": ("leather", "paint_clearcoat", "none"),
    "Organic Material": ("leaf_grass", "none", "none"),
    "Water or Soil": ("mud", "none", "none"),
    "Plastic and Metal": ("plastic", "clearcoat", "none"),
}


# ------------------------- geometry utilities ------------------------- #
def parse_dimension(value: str) -> np.ndarray:
    """Parse dimension string like '10*20*30' (cm) to meters."""
    if not value:
        return np.ones(3, dtype=float)
    parts = [chunk.strip() for chunk in value.lower().split("*") if chunk.strip()]
    if len(parts) != 3:
        return np.ones(3, dtype=float)
    dim_cm = np.array([float(p) for p in parts], dtype=float)
    return dim_cm / 100.0


def read_obj_bounds(path: Path) -> Tuple[np.ndarray, np.ndarray] | None:
    """Compute axis-aligned bounds from an OBJ file."""
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
        return None
    return mins, maxs


def aggregate_bounds(object_id: str, part_labels: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Union the bounds across all part OBJ files of an object."""
    global_min = np.array([np.inf, np.inf, np.inf], dtype=float)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    found = False
    part_dir = MESH_PATH / object_id / "objs"

    for label in part_labels:
        obj_path = part_dir / f"{label}.obj"
        bounds = read_obj_bounds(obj_path)
        if bounds is None:
            continue
        mins, maxs = bounds
        global_min = np.minimum(global_min, mins)
        global_max = np.maximum(global_max, maxs)
        found = True

    if not found:
        # Fallback to unit box if files are missing
        global_min = np.zeros(3, dtype=float)
        global_max = np.ones(3, dtype=float)
    return global_min, global_max


def safe_scale(dimension: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Compute per-axis scale from target dimension and current size."""
    scale = np.ones(3, dtype=float)
    for axis in range(3):
        if size[axis] <= 1e-6 or dimension[axis] <= 0:
            scale[axis] = 1.0
        else:
            raw = dimension[axis] / size[axis]
            scale[axis] = max(raw * SCALE_RELAXATION, 1e-3)
    return scale


def align_dimension_to_size(dimension: np.ndarray, size: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Permute dimension axes so the largest declared dimensions map to the largest mesh extents.

    The Dataset dimensions are not guaranteed to follow the mesh's XYZ ordering. By picking the
    permutation that makes the per-axis scale factors as similar as possible, we avoid extreme
    squashing/stretching caused by axis misalignment.
    """
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
        score = log_scale.ptp()  # minimize spread between per-axis scales
        if score < best_score:
            best_score = score
            best_perm = perm

    return dim[list(best_perm)], best_perm


# ------------------------- IO helpers ------------------------- #
def read_object_json(file_name: str) -> Dict[str, object]:
    """Load per-object metadata (name, dimension, parts, materials)."""
    file_path = JSON_PATH / file_name
    with file_path.open("r") as fh:
        data = json.load(fh)

    info: Dict[str, object] = {
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
                "part_density": part.get("density"),
            }
        )
    return info


def ensure_sim_app():
    """Start Kit/Isaac so pxr/omni modules are available."""
    global _SIM_APP
    if _SIM_APP is None:
        from isaacsim import SimulationApp  # type: ignore

        _SIM_APP = SimulationApp({"headless": True})
    return _SIM_APP


# ------------------------- builders ------------------------- #
def merge_parts_to_obj(object_id: str, scale: np.ndarray, part_labels: List[int], out_path: Path) -> bool:
    """Load, scale, and merge part OBJ meshes into a single OBJ."""
    verts_all: List[np.ndarray] = []
    faces_all: List[np.ndarray] = []
    vert_offset = 0

    part_dir = MESH_PATH / object_id / "objs"
    for label in part_labels:
        obj_path = part_dir / f"{label}.obj"
        if not obj_path.exists():
            print(f"[Warn] Missing part OBJ: {obj_path}")
            continue
        mesh = trimesh.load(obj_path, force="mesh", skip_materials=True)
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"[Warn] Non-mesh content in {obj_path}, skipping.")
            continue
        mesh.apply_scale(scale)
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        verts_all.append(verts)
        faces_all.append(faces + vert_offset)
        vert_offset += len(verts)

    if not verts_all:
        print(f"[Error] No valid parts for {object_id}")
        return False

    merged = trimesh.Trimesh(vertices=np.vstack(verts_all), faces=np.vstack(faces_all), process=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.export(out_path)
    return True

def load_scaled_parts(object_id: str, scale: np.ndarray, parts_meta: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Load, scale, and collect per-part mesh data."""
    part_dir = MESH_PATH / object_id / "objs"
    parts: List[Dict[str, object]] = []
    for part in parts_meta:
        label = part.get("part_id", -1)
        if label == -1:
            continue
        obj_path = part_dir / f"{label}.obj"
        if not obj_path.exists():
            print(f"[Warn] Missing part OBJ: {obj_path}")
            continue
        mesh = trimesh.load(obj_path, force="mesh", skip_materials=True)
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"[Warn] Non-mesh content in {obj_path}, skipping.")
            continue
        mesh.apply_scale(scale)
        parts.append(
            {
                "label": label,
                "name": part.get("part_name", f"part_{label}"),
                "material": part.get("part_material", "Plastic"),
                "density": part.get("part_density"),
                "vertices": np.asarray(mesh.vertices),
                "faces": np.asarray(mesh.faces, dtype=np.int64),
            }
        )
    return parts

def merge_parts_data(parts: List[Dict[str, object]]) -> trimesh.Trimesh:
    """Merge already scaled part meshes into a single Trimesh."""
    verts_all: List[np.ndarray] = []
    faces_all: List[np.ndarray] = []
    vert_offset = 0
    for part in parts:
        verts = np.asarray(part["vertices"])
        faces = np.asarray(part["faces"], dtype=np.int64)
        verts_all.append(verts)
        faces_all.append(faces + vert_offset)
        vert_offset += len(verts)
    return trimesh.Trimesh(vertices=np.vstack(verts_all), faces=np.vstack(faces_all), process=False)


def load_material_properties() -> Dict[str, Dict[str, float]]:
    """Load material friction table once."""
    global _MATERIAL_PROPS
    if _MATERIAL_PROPS is None:
        try:
            with MATERIAL_PROPS_PATH.open("r") as fh:
                _MATERIAL_PROPS = json.load(fh)
        except FileNotFoundError:
            _MATERIAL_PROPS = {}
    return _MATERIAL_PROPS


def get_material_friction(material_name: str, default_friction: float) -> float:
    props = load_material_properties()
    entry = props.get(material_name, {})
    friction = entry.get("friction")
    sigma = entry.get("sigma")
    if friction is None:
        print(f"[Error] Missing friction for material '{material_name}', using default {default_friction}.")
        return default_friction
    else:
        value = np.random.normal(friction, sigma) if sigma is not None else friction
        return max(0.0, min(1.5, value))


def parse_density(value: object, default_density: float) -> float:
    """Parse density strings like '1.2 g/cm^3' to kg/m^3."""
    if value is None:
        return default_density
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        num = None
        for token in text.replace("/", " ").replace("*", " ").split():
            try:
                num = float(token)
                break
            except ValueError:
                continue
        if num is None:
            return default_density
        if "g" in text and "cm" in text:
            return num * 1000.0
        if "kg" in text and ("m" in text):
            return num
        return num
    return default_density


def write_usd_from_parts(
    object_id: str,
    parts: List[Dict[str, object]],
    scale: np.ndarray,
    out_path: Path,
    default_friction: float,
    default_density: float,
    collision_approx: str,
) -> None:
    """Create USD with per-part meshes, visual + non-visual + physics materials."""
    ensure_sim_app()
    from isaacsim.sensors.rtx import apply_nonvisual_material  # type: ignore
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, PhysxSchema  # type: ignore

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Align with mesh_utils.convert_mesh_to_usd: root prim is the object itself
    obj_xform = UsdGeom.Xform.Define(stage, Sdf.Path("/object"))
    # Treat whole object as one rigid body; collisions on child meshes
    UsdPhysics.RigidBodyAPI.Apply(obj_xform.GetPrim())
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(obj_xform.GetPrim())
    physx_body.GetEnableCCDAttr().Set(True)
    # Disable gravity via PhysX API attribute to match usd_dataset behavior
    physx_body.GetDisableGravityAttr().Set(True)
    obj_xform.AddRotateXYZOp().Set((90.0, 0.0, 0.0))

    # Material caches to avoid duplicates
    visual_cache: Dict[str, UsdShade.Material] = {}

    # Keep all authored paths under the object prim so they relocate correctly when referenced
    obj_path_str = obj_xform.GetPath().pathString
    looks_scope_path = f"{obj_path_str}/Looks"
    phys_scope_path = f"{obj_path_str}/PhysicsMaterials"

    looks_scope = stage.GetPrimAtPath(looks_scope_path)
    if not looks_scope:
        looks_scope = stage.DefinePrim(looks_scope_path, "Scope")
    phys_scope = stage.GetPrimAtPath(phys_scope_path)
    if not phys_scope:
        phys_scope = stage.DefinePrim(phys_scope_path, "Scope")

    for part in parts:
        label = part["label"]
        material_name = str(part.get("material", "Plastic"))
        verts = np.asarray(part["vertices"])
        faces = np.asarray(part["faces"], dtype=np.int64)

        mesh_path = Sdf.Path(f"/object/part_{label}")
        mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)
        mesh_prim.CreatePointsAttr(verts.tolist())
        mesh_prim.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh_prim.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
        mesh_prim.CreateSubdivisionSchemeAttr("none")

        mesh_prim_prim = mesh_prim.GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh_prim_prim)
        PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim_prim)
        UsdPhysics.MeshCollisionAPI.Apply(mesh_prim_prim)
        if collision_approx == "convexDecomposition":
            PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh_prim_prim)
        elif collision_approx == "convexHull":
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh_prim_prim)
        mesh_prim_prim.GetAttribute("physics:approximation").Set(collision_approx)

        # Visual material (MDL) per material_name
        mat_token = material_name.replace(" ", "_")
        if mat_token not in visual_cache:
            mdl_name, mdl_path = ISAAC_MATERIALS_MDL.get(material_name, ("Plastic", "Plastics/Plastic.mdl"))
            visual_mat_path = Sdf.Path(f"{looks_scope_path}/{mat_token}")
            visual_mat = UsdShade.Material.Define(stage, visual_mat_path)
            shader = UsdShade.Shader.Define(stage, visual_mat_path.AppendPath("Shader"))
            shader.CreateIdAttr("mdlMaterial")
            shader.CreateInput("mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(str((MATERIAL_DIR / mdl_path).as_posix()))
            shader.CreateInput("mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.String).Set(mdl_name)
            shader_out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
            visual_mat.CreateSurfaceOutput().ConnectToSource(shader_out)
            visual_cache[mat_token] = visual_mat

            nv = DATASET_TO_ISAAC.get(material_name, ("plastic", "paint", "none"))
            apply_nonvisual_material(visual_mat.GetPrim(), nv[0], nv[1], nv[2])

        UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim()).Bind(visual_cache[mat_token])

        # Physics material per material_name
        # Physics material per part (density can vary), follow GraspDataGen style
        part_friction = get_material_friction(material_name, default_friction)
        part_density = parse_density(part.get("density"), default_density)
        phys_mat_path = Sdf.Path(f"{phys_scope_path}/{mat_token}_part{label}")
        phys_mat = UsdShade.Material.Define(stage, phys_mat_path)
        mat_api = UsdPhysics.MaterialAPI.Apply(phys_mat.GetPrim())
        mat_api.CreateStaticFrictionAttr(part_friction)
        mat_api.CreateDynamicFrictionAttr(part_friction)
        mat_api.CreateDensityAttr(part_density)
        UsdShade.MaterialBindingAPI.Apply(mesh_prim_prim).Bind(
            phys_mat, 
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics")

        mesh_prim.GetPrim().CreateAttribute("customData:part_id", Sdf.ValueTypeNames.Int).Set(int(label))
        mesh_prim.GetPrim().CreateAttribute("customData:part_material", Sdf.ValueTypeNames.String).Set(material_name)

    # Metadata for traceability on root
    obj_xform.GetPrim().CreateAttribute("customData:file_id", Sdf.ValueTypeNames.String).Set(object_id)
    obj_xform.GetPrim().CreateAttribute("customData:scale", Sdf.ValueTypeNames.Double3).Set(tuple(scale.tolist()))

    stage.SetDefaultPrim(obj_xform.GetPrim())
    stage.GetRootLayer().Save()


def process_one(object_id: str, friction: float, density: float, collision_approx: str) -> None:
    print(f"[Step] Reading metadata for {object_id}")
    meta = read_object_json(f"{object_id}.json")
    parts_meta = meta["parts"]
    part_labels = [p["part_id"] for p in parts_meta if p["part_id"] != -1]
    if not part_labels:
        print(f"[Skip] No parts listed for {object_id}")
        return

    # print(f"[Step] Aggregating bounds for parts: {part_labels}")
    bbox_min, bbox_max = aggregate_bounds(object_id, part_labels)
    # print(f"[Step] Computed bounds min {bbox_min.round(4)} m, max {bbox_max.round(4)} m")
    size = bbox_max - bbox_min
    dim_raw = parse_dimension(meta.get("dimension", ""))
    dim, perm = align_dimension_to_size(dim_raw, size)
    # print(f"[Step] Parsed dimension {dim_raw.round(4)} m, reordered to {dim.round(4)} m for axis order {perm}, current size {size.round(4)} m")
    scale = safe_scale(dim, size)
    # print(f"[Step] Computed scale {scale.round(4)} from dim {dim} and size {size}")

    obj_out = OUTPUT_OBJ / f"{object_id}.obj"
    print(f"[Step] Merging parts to OBJ -> {obj_out}")
    parts = load_scaled_parts(object_id, scale, parts_meta)
    if not parts:
        print(f"[Error] No valid parts loaded for {object_id}")
        return
    merged = merge_parts_data(parts)
    obj_out.parent.mkdir(parents=True, exist_ok=True)
    merged.export(obj_out)

    print(f"[Step] Writing USD with per-part meshes -> {OUTPUT_USD}")
    usd_out = OUTPUT_USD / f"{object_id}.usd"
    write_usd_from_parts(object_id, parts, scale, usd_out, friction, density, collision_approx=collision_approx)
    print(f"[OK] {object_id}: OBJ->{obj_out}, USD->{usd_out}, scale={scale.round(4)}")


# ------------------------- CLI ------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged OBJ and USD assets from Dataset parts.")
    parser.add_argument("--ids", nargs="*", help="Object ids to process (default: all JSON files).")
    parser.add_argument("--friction", type=float, default=1.0, help="Static/dynamic friction to write into USD.")
    parser.add_argument("--density", type=float, default=1000.0, help="Density (kg/m^3) to write into USD.")
    parser.add_argument(
        "--collision-approx",
        choices=["convexDecomposition", "convexHull"],
        default="convexDecomposition",
        help="Collision approximation per part (use convexDecomposition for concave shapes).",
    )
    parser.add_argument("--root", type=Path, default=None, help="Dataset root directory (default: /datasets).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root directory (default: <root>/converted).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths(dataset_root=args.root, output_root=args.output)
    OUTPUT_OBJ.mkdir(parents=True, exist_ok=True)
    OUTPUT_USD.mkdir(parents=True, exist_ok=True)

    # Ensure the SimulationApp is torn down when we finish.
    global _SIM_APP

    ids = args.ids
    print(f"[Info] Processing {len(ids) if ids else 'all'} objects...")
    if not ids:
        ids = sorted(p.stem for p in JSON_PATH.glob("*.json"))

    for object_id in ids:
        print(f"[Info] Start processing {object_id}")
        try:
            process_one(object_id, friction=args.friction, density=args.density, collision_approx=args.collision_approx)
        except Exception as exc:
            print(f"[Error] {object_id}: {exc}")
    if _SIM_APP is not None:
        _SIM_APP.close()


if __name__ == "__main__":
    main()
