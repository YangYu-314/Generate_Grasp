import argparse
import os
import sys
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import matplotlib.pyplot as plt
import numpy as np
import omni
import yaml
from pxr import UsdGeom

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .utils.loader import StageLoader
from .utils.sensors import SensorBuilder


def create_environment(config_path):
    """Create the simulation environment for a given configuration file."""
    stage = omni.usd.get_context().get_stage()

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    stage_loader = StageLoader(
        stage,
        assets_directory="/scratch2/yangyu/workspace/Dataset/mesh/",
        assets_config_directory="/scratch2/yangyu/workspace/Dataset/json/",
        material_directory="/scratch2/yangyu/workspace/Dataset/material/",
        stage_config=config
    )

    # stage_loader.clear_stage()
    stage_loader.build_scene()
    print(f"[Simulation] Success build scene for config {config_path}.")

    sensor_builder = SensorBuilder(
        11,
        13,
        None,
        None,
        0.5,
        0.5,
    )

    sensor_builder.build_sensors(stage)
    print("[Simulation] Success build sensors.")

    return stage, sensor_builder


def get_world_matrix(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        print(f"[Simulation][Warning] Prim not found at path: {prim_path}")
        return None
    xf = UsdGeom.Xformable(prim)
    return xf.ComputeLocalToWorldTransform(0.0)


def mesh_points_to_world(stage, prim, points):
    """Convert mesh-local vertices to world coordinates."""
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    world_mat = get_world_matrix(stage, prim.GetPath())
    world_pts = hom @ world_mat
    return world_pts[:, :3]


def extract_mesh(stage, prim):
    mesh = UsdGeom.Mesh(prim)
    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

    tris = []
    idx = 0
    for cnt in counts:
        if cnt == 3:
            tris.append(indices[idx:idx + 3])
        elif cnt == 4:
            quad = indices[idx:idx + 4]
            tris.append([quad[0], quad[1], quad[2]])
            tris.append([quad[0], quad[2], quad[3]])
        else:
            raise ValueError(f"Unsupported face vertex count: {cnt}")
        idx += cnt

    tris = np.array(tris, dtype=np.int32)
    return mesh_points_to_world(stage, prim, points), tris


def split_transform(mat):
    """Return matrix as-is and extract per-axis scale without normalizing rotation."""
    R = mat[:3, :3]
    scale = np.linalg.norm(R, axis=0)
    return mat, scale


def semantic_voxel_sample(points_with_labels, voxel_size=0.005):
    """Down-sample points while preserving per-part semantics."""
    if len(points_with_labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    xyz = points_with_labels[:, :3]
    labels = points_with_labels[:, 3:].astype(np.int32)

    voxel_coords = np.floor(xyz / voxel_size).astype(np.int32)
    buckets = {}

    for idx, coord in enumerate(voxel_coords):
        key = tuple(coord)
        buckets.setdefault(key, []).append(idx)

    sampled = []
    for idxs in buckets.values():
        pts = xyz[idxs]
        lbls = labels[idxs]
        uniq, inv = np.unique(lbls, axis=0, return_inverse=True)
        for j, label in enumerate(uniq):
            pts_subset = pts[inv == j]
            centroid = pts_subset.mean(axis=0)
            sampled.append(np.concatenate([centroid, label]))

    return np.array(sampled, dtype=np.float32)


def collect_usd(stage, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    usd_path = os.path.join(save_dir, "scene.usd")
    omni.usd.get_context().save_as_stage(usd_path)
    print("[Simulation] USD saved to", usd_path)


def collect_camera_images(sensor_builder, save_dir):
    images = sensor_builder.get_camera_outputs()
    for idx, img in enumerate(images):
        plt.imsave(os.path.join(save_dir, f"camera_{idx}.png"), img)
    print("[Simulation] Camera images saved.")


def collect_camera_transforms(sensor_builder, stage, save_dir):
    """Save camera extrinsics (local-to-world) and intrinsics for each camera."""
    extr, intr = sensor_builder.get_camera_transforms()
    np.savez_compressed(
        os.path.join(save_dir, "camera_transforms.npz"),
        extrinsics=extr,
        intrinsics=intr,
    )
    print("[Simulation] Camera transforms saved.")


def collect_lidar_points(sensor_builder, stage, save_dir):
    all_pts, all_intensities = [], []
    for _ in range(50):
        simulation_app.update()
        pts, intensities = sensor_builder.get_lidar_outputs(stage)
        if len(pts):
            all_pts.append(pts)
            all_intensities.append(intensities)

    if all_pts:
        all_pts = np.concatenate(all_pts, axis=0)
        all_intensities = np.concatenate(all_intensities, axis=0)
    else:
        print("[Simulation] No LiDAR points collected.")
        return

    mask = (
        (all_pts[:, 0] >= -2) & (all_pts[:, 0] <= 2) &
        (all_pts[:, 1] >= -2) & (all_pts[:, 1] <= 2) &
        (all_pts[:, 2] > 0.004) & (all_pts[:, 2] <= 1.0)
    )
    all_pts = all_pts[mask]
    all_intensities = all_intensities[mask]

    np.savez_compressed(os.path.join(save_dir, "lidar_points.npz"), points=all_pts, intensities=all_intensities)
    print("[Simulation] LiDAR point clouds saved.")

    if len(all_pts):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2], c=all_intensities, cmap="viridis")
        ax.set_zlim3d([0, 1.0])
        plt.savefig(os.path.join(save_dir, "lidar_output.png"))
        print("[Simulation] Lidar visualization saved.")


def collect_mesh_data(stage, save_dir):
    mesh_verts, mesh_faces, obj_labels, part_labels = [], [], [], []
    vert_offset = 0
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith("/World/Assets/object_"):
            continue
        if "part_" not in path or prim.GetTypeName() != "Mesh":
            continue
        object_token = path.split("object_")[1].split("/")[0]
        if "_obj" in object_token:
            try:
                object_id = int(object_token.split("_obj")[1])
            except Exception:
                object_id = int(object_token.split("_obj")[0])
        else:
            object_id = int(object_token)
        part_id = int(path.split("part_")[1].split("_")[0])

        verts, faces = extract_mesh(stage, prim)
        mesh_verts.append(verts)
        # Offset face indices to account for already appended vertices.
        mesh_faces.append(faces + vert_offset)
        obj_labels.append(np.full(len(verts), object_id, dtype=np.int32))
        part_labels.append(np.full(len(verts), part_id, dtype=np.int32))
        vert_offset += len(verts)

    if not mesh_verts:
        print("[Simulation][Warning] No mesh data collected from /World/Assets.")
        return None

    verts_all = np.concatenate(mesh_verts, axis=0)
    faces_all = np.concatenate(mesh_faces, axis=0)
    obj_all = np.concatenate(obj_labels, axis=0)
    part_all = np.concatenate(part_labels, axis=0)

    np.savez_compressed(
        os.path.join(save_dir, "scene_mesh.npz"),
        verts=verts_all,
        faces=faces_all,
        obj=obj_all,
        part=part_all,
    )
    print("[Simulation] Scene mesh saved.")

    semantic_points = np.concatenate(
        [verts_all, obj_all[:, None].astype(np.float32), part_all[:, None].astype(np.float32)],
        axis=1,
    )
    sampled = semantic_voxel_sample(semantic_points, voxel_size=0.005)
    np.savez_compressed(
        os.path.join(save_dir, "scene_mesh_uniform.npz"),
        points=sampled[:, :3],
        obj=sampled[:, 3].astype(np.int32),
        part=sampled[:, 4].astype(np.int32),
    )
    print("[Simulation] Scene mesh voxel samples saved.")


def collect_object_transforms(stage, save_dir):
    transforms, scales, ids = [], [], []
    assets_root = stage.GetPrimAtPath("/World/Assets")
    if not assets_root:
        print("[Simulation][Warning] /World/Assets prim does not exist; cannot record transforms.")
        return

    for child in assets_root.GetChildren():
        name = child.GetName()
        if not name.startswith("object_"):
            continue
        token = name.split("object_")[1]
        if "_obj" in token:
            try:
                obj_id = int(token.split("_obj")[1])
            except Exception:
                obj_id = None
        else:
            obj_id = None
        if obj_id is None:
            try:
                obj_id = int(token)
            except (IndexError, ValueError):
                continue

        mat = np.array(UsdGeom.Xformable(child).ComputeLocalToWorldTransform(0.0))
        mat_out, scale = split_transform(mat)
        ids.append(obj_id)
        transforms.append(mat_out.astype(np.float32))
        scales.append(scale.astype(np.float32))

    if not transforms:
        print("[Simulation][Warning] No object transforms found under /World/Assets.")
        return

    np.savez_compressed(
        os.path.join(save_dir, "object_transforms.npz"),
        object_ids=np.array(ids, dtype=np.int32),
        transforms=np.stack(transforms, axis=0),
        scales=np.stack(scales, axis=0),
    )
    print("[Simulation] Object transforms saved.")


def run_simulation(config_path: Path, output_root: Path):
    """Run a single scene using the specified config."""
    if not config_path.exists():
        print(f"[Simulation][Error] Config file not found: {config_path}")
        simulation_app.close()
        return

    print(f"[Simulation] Starting scene for configuration: {config_path}")
    timeline = omni.timeline.get_timeline_interface()

    stage, sensor_builder = create_environment(config_path)

    timeline.play()
    for _ in range(240):
        simulation_app.update()
    print("[Simulation] Simulator stabilized.")

    save_dir = output_root / config_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    collect_camera_images(sensor_builder, save_dir)
    collect_camera_transforms(sensor_builder, stage, save_dir)
    collect_lidar_points(sensor_builder, stage, save_dir)
    collect_mesh_data(stage, save_dir)
    collect_object_transforms(stage, save_dir)
    collect_usd(stage, save_dir)
    timeline.stop()

    simulation_app.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single Isaac Sim scene generation.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML config used to build the scene.",
    )
    parser.add_argument(
        "--output-root",
        default=Path("./outputs"),
        type=Path,
        help="Directory where scene outputs will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_simulation(args.config.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
