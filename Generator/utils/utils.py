from pxr import Gf, UsdGeom, Usd
import numpy as np
import re
import random

NUM_OF_OBJECTS = 3041

def check_valid_quat(quat):
    """Check if the given quaternion is valid (i.e., has a non-zero norm and is normalized).

    Args:
        quat (tuple): A tuple representing the quaternion (w, x, y, z).
    
    Returns:
        bool: True if the quaternion is valid, False otherwise.
    """
    print(quat)
    if len(quat) != 4:
        return False
    norm = sum(q ** 2 for q in quat) ** 0.5
    return norm > 1e-6 and abs(norm - 1.0) < 1e-3

def normalize_quat(quat):
    """Normalize the given quaternion.

    Args:
        quat (tuple): A tuple representing the quaternion (w, x, y, z).
    
    Returns:
        tuple: A normalized quaternion.
    """
    norm = sum(q ** 2 for q in quat) ** 0.5
    if norm < 1e-6:
        raise ValueError("Cannot normalize a zero-length quaternion.")
    return tuple(q / norm for q in quat)

def quat_from_euler(yaw, pitch, roll):
    """Convert Euler rotation (in degrees) to a quaternion.
    
    Args:
        yaw (float): Rotation around the Z axis in degrees.
        pitch (float): Rotation around the Y axis in degrees.
        roll (float): Rotation around the X axis in degrees.
    
    Returns:
        tuple: A tuple representing the quaternion (w, x, y, z).
    """
    rot_yaw = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)    # around Z
    rot_pitch = Gf.Rotation(Gf.Vec3d(0, 1, 0), pitch) # around Y
    rot_roll = Gf.Rotation(Gf.Vec3d(1, 0, 0), roll)   # around X
    rot = rot_yaw * rot_pitch * rot_roll
    return rot.GetQuat()

def transform_local_to_world(points_local, lidar_prim):
    """Transform points from local LiDAR frame to world frame.
    
    Args:
        points_local (np.ndarray): Nx3 array of points in the local LiDAR frame.
        lidar_prim (Usd.Prim): The USD prim representing the LiDAR sensor.

    Returns:
        np.ndarray: Nx3 array of points in the world frame.
    """
    if points_local is None:
        return np.zeros((0, 3), dtype=np.float32)

    points_local = np.array(points_local)
    if points_local.ndim == 1:
        size = points_local.size
        if size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if size % 3 == 0:
            points_local = points_local.reshape((-1, 3))
        elif size % 4 == 0:
            # Some configs may include an extra channel; drop it.
            points_local = points_local.reshape((-1, 4))[:, :3]
        else:
            # Unexpected shape; bail out gracefully.
            return np.zeros((0, 3), dtype=np.float32)
    if points_local.shape[0] == 0 or points_local.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)

    xform = UsdGeom.Xformable(lidar_prim)

    # Get the local to world transformation matrix, [[R 0], [t 1]]
    world_mat = np.array(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

    # Convert points to homogeneous coordinates
    n = points_local.shape[0]
    points_h = np.concatenate([points_local, np.ones((n, 1))], axis=1)  # [x y z 1]

    # Multiply points by transformation matrix
    # Multiply points by transformation matrix (row-vector convention)
    points_world_h = points_h @ world_mat
    return points_world_h[:, :3]

def generate_default_setting(num, is_physics=True):
    """Generate default object settings with random positions.
    
    Args:
        num (int): Number of objects to generate.
        is_physics (bool): Whether the objects should have rigid body physics enabled.
    """

    objects = []
    obj_idx = random.sample(range(NUM_OF_OBJECTS), num)
    for i in obj_idx:
        obj_id = i
        position = [
            random.uniform(-0.5, 0.5),
            random.uniform(-0.5, 0.5),
            random.uniform(0.3, 1.0)
        ]
        rotation = [
            0.0, 90.0, 0.0
        ]
        scale = [33, 33, 33]
        objects.append({
            "name": str(obj_id),
            "position": position,
            "rotation": rotation,
            "scale": scale,
            "physics": is_physics
        })
    return objects


def sanitize_name(name: str) -> str:
    """Sanitize a string to be used as a valid USD prim name.
    """
    return re.sub(r'[^A-Za-z0-9_]', '_', name)
