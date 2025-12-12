from __future__ import annotations

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

# Ensure project imports work when launched via `isaaclab.sh -p Generator/usd_dataset.py`
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# from isaacsim import SimulationApp

# config = {
#     "headless": True,
#     "isaac/asset_root/default": "/isaacsim_assets/Assets/Isaac/5.1",
#     "isaac/asset_root/nvidia": "/isaacsim_assets/Assets/Isaac/5.1", 
# }

# # Launch Isaac Sim in headless mode for batch dataset generation
# simulation_app = SimulationApp(config)

from isaacsim import SimulationApp
import carb

# 1) 正常启动 Isaac Sim（不要在这里传 asset_root）
simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting",})

# 2) 用正确的 setting 名称覆盖资产根目录（容器内路径）
LOCAL_ASSET_ROOT = "/isaacsim_assets/Assets/Isaac/5.1"

simulation_app.set_setting(
    "/persistent/isaac/asset_root/default",
    LOCAL_ASSET_ROOT,
)
simulation_app.set_setting(
    "/persistent/isaac/asset_root/nvidia",
    LOCAL_ASSET_ROOT,
)

# 3) 打印确认真的写进去了（这里读设置，不调用 get_assets_root_path，避免立刻抛错）
settings = carb.settings.get_settings()
print("[DEBUG] default setting   =", settings.get("/persistent/isaac/asset_root/default"))
print("[DEBUG] nvidia  setting   =", settings.get("/persistent/isaac/asset_root/nvidia"))

# Use non-interactive backend for headless export
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import omni  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

from isaacsim.sensors.rtx import LidarRtx  # noqa: E402
from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema  # noqa: E402

from Generator.utils.sensors import SensorBuilder  # noqa: E402
from Generator.utils.utils import quat_from_euler  # noqa: E402


MATERIAL_PROPS_PATH = Path(__file__).resolve().parent.parent / "material_properties.json"


def load_material_properties(path: Path) -> Dict[str, Dict[str, float]]:
    try:
        with path.open("r") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        print(f"[Dataset][Warning] material properties file not found: {path}")
        return {}


def load_ids_from_chunk(chunk_path: Path) -> List[int]:
    """Read a chunk JSON (keys like '0.obj') and return sorted unique integer IDs."""
    try:
        with chunk_path.open("r") as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"[Dataset][Error] Failed to read chunk file {chunk_path}: {exc}")
        return []

    keys = data.keys() if isinstance(data, dict) else data
    ids: List[int] = []
    for key in keys:
        name = str(key)
        if name.endswith(".usd") or name.endswith(".obj"):
            name = name.rsplit(".", 1)[0]
        if name.isdigit():
            ids.append(int(name))
    return sorted(set(ids))


def build_fixed_lidar_positions(distance: float):
    """Create a fixed six-direction LiDAR layout at a constant radius."""
    return {
        "lidar_front": ((distance, 0.0, 0.0), quat_from_euler(180.0, 0.0, 0.0)),
        "lidar_back": ((-distance, 0.0, 0.0), quat_from_euler(0.0, 0.0, 0.0)),
        "lidar_left": ((0.0, distance, 0.0), quat_from_euler(90.0, 0.0, 0.0)),
        "lidar_right": ((0.0, -distance, 0.0), quat_from_euler(-90.0, 0.0, 0.0)),
        "lidar_up": ((0.0, 0.0, distance), quat_from_euler(0.0, 90.0, 0.0)),
        "lidar_down": ((0.0, 0.0, -distance), quat_from_euler(0.0, -90.0, 0.0)),
    }


def reset_stage(stage) -> None:
    """Remove previous prims to start a clean stage."""
    try:
        rep.orchestrator.get_instance().reset()
    except Exception as exc:
        print(f"[Dataset][Warning] Failed to reset replicator orchestrator: {exc}")

    world = stage.GetPrimAtPath("/World")
    if world.IsValid():
        for child in list(world.GetChildren()):
            stage.RemovePrim(child.GetPath())
    else:
        stage.DefinePrim("/World", "Xform")

    looks = stage.GetPrimAtPath("/Looks")
    if looks.IsValid():
        for child in list(looks.GetChildren()):
            stage.RemovePrim(child.GetPath())
    else:
        stage.DefinePrim("/Looks", "Scope")

    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def disable_gravity(stage):
    """Mimic GraspDataGen: no global gravity, disable gravity on object root."""
    # Global gravity off
    phys_scene = UsdPhysics.Scene.Get(stage, "/World/physicsScene")
    if not phys_scene:
        phys_scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    phys_scene.CreateGravityDirectionAttr().Set((0.0, 0.0, 0.0))
    phys_scene.CreateGravityMagnitudeAttr().Set(0.0)
    PhysxSchema.PhysxSceneAPI.Apply(phys_scene.GetPrim())

    # Disable gravity on the referenced object root (Xform is fine for this attr)
    obj_prim = stage.GetPrimAtPath("/World/Assets/object")
    if obj_prim and obj_prim.IsValid():
        attr = obj_prim.GetAttribute("physx:disableGravity")
        if not attr:
            attr = obj_prim.CreateAttribute("physx:disableGravity", Sdf.ValueTypeNames.Bool)
        attr.Set(True)


def reference_usd(stage, usd_path: Path) -> str:
    """Reference a USD asset under /World/Assets/object and return its prim path (no rescale/extra transforms)."""
    assets_root = stage.DefinePrim("/World/Assets", "Xform")
    object_prim = stage.DefinePrim("/World/Assets/object", "Xform")
    omni.kit.commands.execute(
        "AddReferenceCommand",
        prim_path=str(object_prim.GetPath()),
        stage=stage,
        reference=Sdf.Reference(assetPath=str(usd_path)),
    )
    return str(object_prim.GetPath())


def decode_object_paths(object_ids: np.ndarray, stable_maps: List[np.ndarray]) -> List[str | None]:
    """Map RTX objectId buffer to prim paths using StableIdMap."""
    stable_bytes = None
    for entry in stable_maps:
        if entry is None:
            continue
        try:
            stable_bytes = entry.tobytes()
            break
        except Exception:
            continue

    if stable_bytes is None or object_ids.size == 0:
        return [None] * len(object_ids)

    stable_map = LidarRtx.decode_stable_id_mapping(stable_bytes)
    decoded_ids = LidarRtx.get_object_ids(object_ids)
    return [stable_map.get(obj_id) for obj_id in decoded_ids]


def get_physics_properties(prim, material_props: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    """Return friction and density for a prim using material_properties first, then physics material."""
    material_name = None
    attr = prim.GetAttribute("customData:part_material")
    if attr and attr.HasAuthoredValue():
        material_name = attr.Get()

    friction = None
    density = None
    if material_name:
        entry = material_props.get(str(material_name), {})
        if "friction" in entry:
            try:
                friction = float(entry["friction"])
            except Exception:
                pass
        if "density" in entry:
            try:
                density = float(entry["density"])
            except Exception:
                pass

    # Fallback to physics material bound on the prim
    binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if binding:
        material, _ = binding
        mat_api = UsdPhysics.MaterialAPI.Get(prim.GetStage(), material.GetPrim().GetPath())
        if mat_api:
            if friction is None:
                friction = mat_api.GetStaticFrictionAttr().Get()
            if density is None:
                density = mat_api.GetDensityAttr().Get()

    if friction is None:
        friction = 0.0
    if density is None:
        density = 0.0
    return float(friction), float(density)


def build_material_cache(stage, material_props: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    cache: Dict[str, Tuple[float, float]] = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        friction, density = get_physics_properties(prim, material_props)
        mat_attr = prim.GetAttribute("customData:part_material")
        mat_name = mat_attr.Get() if mat_attr and mat_attr.HasAuthoredValue() else None
        binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        vis_path = binding[0].GetPath() if binding else None
        print(f"[MaterialCache] {prim.GetPath()} mat={mat_name} vis={vis_path} friction={friction} density={density}")
        cache[prim.GetPath().pathString] = (friction, density)
    return cache


def collect_lidar_data(
    stage,
    sensor_builder: SensorBuilder,
    frames: int,
    warmup: int,
) -> Tuple[np.ndarray, np.ndarray]:
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(warmup):
        simulation_app.update()

    points, intensities = [], []
    # material_ids, object_ids, stable_maps = [], [], [], []
    for i in range(frames):
        simulation_app.update()
        # pts, inten, mat_id, obj_id, stable = sensor_builder.get_lidar_outputs(stage, return_metadata=True)
        pts, inten = sensor_builder.get_lidar_outputs(stage, return_metadata=False)
        print(f"Frame {i+1}/{frames}: Collected {len(pts)} LiDAR points")
        # print(mat_id)
        # print(obj_id)
        if len(pts):
            points.append(pts)
            intensities.append(inten)
            # material_ids.append(mat_id)
            # object_ids.append(obj_id)
            # stable_maps.extend(stable)

    timeline.stop()

    if not points:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            # np.zeros((0,), dtype=np.uint8),
            # np.zeros((0,), dtype=np.uint8),
            # stable_maps,
        )

    all_points = np.concatenate(points, axis=0).astype(np.float32)
    all_intensity = np.concatenate(intensities, axis=0).astype(np.float32)

    # all_intensity = (all_intensity - all_intensity.min()) / (all_intensity.max() - all_intensity.min())
    # all_mat_ids = np.concatenate(material_ids, axis=0)
    # all_obj_ids = np.concatenate(object_ids, axis=0)
    return all_points, all_intensity


def save_arrays(output_dir: Path, stem: str, points: np.ndarray, intensity: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{stem}.npy", points.astype(np.float32))
    np.save(output_dir / f"{stem}_intensity.npy", np.concatenate([points, intensity[:, None]], axis=1).astype(np.float32))


def save_visualizations(
    output_dir: Path,
    stem: str,
    points: np.ndarray,
    intensity: np.ndarray,
    friction: np.ndarray | None = None,
    density: np.ndarray | None = None,
) -> None:
    """Save scatter visualizations colored by intensity (friction/density optional)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(points) == 0:
        return

    def _plot(values, title, fname):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values, s=1, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.07)
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)
        ax.set_zlim(-0.1, 0.1)
        plt.tight_layout()
        fig.savefig(output_dir / fname, dpi=200)
        plt.close(fig)

    _plot(intensity, "LiDAR Intensity", f"{stem}_viz_intensity.png")
    if friction is not None and len(friction):
        _plot(friction, "Friction", f"{stem}_viz_friction.png")
    if density is not None and len(density):
        _plot(density, "Density", f"{stem}_viz_density.png")


def process_usd_file(
    usd_path: Path,
    output_root: Path,
    args,
) -> None:
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    reset_stage(stage)
    reference_usd(stage, usd_path)
    # Ensure basic lighting so camera renders are visible.
    # if not stage.GetPrimAtPath("/World/KeyLight"):
    #     omni.kit.commands.execute(
    #         "CreatePrim",
    #         prim_type="DistantLight",
    #         prim_path="/World/KeyLight",
    #         attributes={"inputs:intensity": 5000.0, "inputs:angle": 2.0},
    #     )

    disable_gravity(stage)

    # Debug: print per-mesh material bindings and physics properties.
    # _ = build_material_cache(stage, material_props)

    # fixed_lidar_positions = build_fixed_lidar_positions(args.lidar_distance)

    sensor_builder = SensorBuilder(
        # num_lidars=len(fixed_lidar_positions),
        num_lidars=args.lidars,
        num_cameras=args.cameras,
        # lidar_positions=fixed_lidar_positions,
        camera_positions=None,
        distance_lidar=args.lidar_distance,
        distance_camera=args.camera_distance,
        include_bottom_lidar=args.include_bottom,
    )
    sensor_builder.build_sensors(stage)

    points, intensities = collect_lidar_data(stage, sensor_builder, args.frames, args.warmup)
    if len(points) == 0:
        print(f"[Dataset][Error] No LiDAR returns for {usd_path}")
        # Propagate failure so batch runner can flag this ID.
        raise RuntimeError(f"No LiDAR returns for {usd_path}")

    object_out_dir = output_root / usd_path.stem
    object_out_dir.mkdir(parents=True, exist_ok=True)
    save_arrays(object_out_dir, usd_path.stem, points, intensities)
    if args.visualize:
        save_visualizations(object_out_dir, usd_path.stem, points, intensities)
    # Copy source USD and sibling OBJ into the same folder for provenance.
    try:
        shutil.copy2(usd_path, object_out_dir / usd_path.name)
    except Exception as exc:
        print(f"[Dataset][Warning] Failed to copy USD {usd_path} -> {object_out_dir}: {exc}")
    obj_candidate = usd_path.with_suffix(".obj")
    if obj_candidate.exists():
        try:
            shutil.copy2(obj_candidate, object_out_dir / obj_candidate.name)
        except Exception as exc:
            print(f"[Dataset][Warning] Failed to copy OBJ {obj_candidate} -> {object_out_dir}: {exc}")
    else:
        print(f"[Dataset][Warning] OBJ not found for {usd_path.stem}: {obj_candidate}")

    print(f"[Dataset] Saved {usd_path.stem} arrays to {object_out_dir}")


def gather_usd_files(root: Path, ids: List[int] | None = None) -> List[Path]:
    # Single-USD mode: expect a file path; directories are rejected to avoid unintended batches.
    if not root.is_file():
        print(f"[Dataset][Error] --usd must point to a single USD file, got directory: {root}")
        return []
    return [root]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LiDAR datasets (xyz, xyz+intensity) for USD assets.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--usd", type=Path, help="Path to a single USD file (Single mode).")
    group.add_argument("--root", type=Path, help="Dataset root directory containing a 'usd' folder (Batch mode).")

    parser.add_argument("--ids", type=str, nargs="+", help="List of object IDs to process (Batch mode only).")    
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root output directory; results will be stored in <output-dir>/<usd_stem>/ (defaults to USD parent).",
    )
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to accumulate after warmup.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of frames to simulate before recording.")
    parser.add_argument("--lidars", type=int, default=12, help="Number of LiDARs to spawn (top and bottom included in this count if bottom is enabled).")
    parser.add_argument("--lidar-distance", type=float, default=0.6 , help="Radius used to position LiDARs around the object.")
    parser.add_argument("--cameras", type=int, default=0, help="Number of cameras to spawn for visual inspection.")
    parser.add_argument("--camera-distance", type=float, default=0.5, help="Radius used to position cameras around the object.")
    parser.add_argument("--include-bottom", action="store_true", dest="include_bottom", help="Place one LiDAR directly below the object looking up.")
    parser.add_argument("--no-bottom", action="store_false", dest="include_bottom", help="Disable the bottom LiDAR.")
    parser.set_defaults(include_bottom=True)
    # parser.add_argument("--material-props", type=Path, default=MATERIAL_PROPS_PATH, help="Path to material_properties.json.")
    parser.add_argument("--visualize", action="store_true", help="Export PNG visualizations for intensity.")
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = []

    if args.usd:
        if not args.usd.exists():
            print(f"[Error] File not found: {args.usd}")
            simulation_app.close()
            return
        tasks.append(args.usd)
    
    elif args.root and args.ids:
        usd_dir = args.root / "usd"
        for obj_id in args.ids:
            file_name = f"{obj_id}.usd"
            usd_path = usd_dir / file_name
            if usd_path.exists():
                tasks.append(usd_path)
            else:
                print(f"[Warning] USD file not found: {usd_path}")

    if not tasks:
        print("[Error] No valid tasks found.")
        simulation_app.close()
        return

    print(f"[Main] Found {len(tasks)} tasks to process.")

    base_out_dir = args.output_dir if args.output_dir else tasks[0].parent

    for usd_path in tasks:
        try:
            process_usd_file(usd_path, base_out_dir, args)
        except Exception as e:
            print(f"[Error] Failed processing {usd_path.name}: {e}")
            continue

    simulation_app.close()


if __name__ == "__main__":
    main()
