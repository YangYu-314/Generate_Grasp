import omni
import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera
from .utils import check_valid_quat, quat_from_euler, transform_local_to_world
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
from pxr import UsdGeom, Usd, Sdf


class SensorBuilder:
    """Utility class for creating and managing sensors in the simulation environment.

    This class provides methods to create and manage various types of sensors,
    such as LiDARs, cameras in the Isaac Sim environment.

    Attributes:
        Num_LiDARs (int): The number of LiDAR sensors to create, default is 5.
        Num_Cameras (int): The number of camera sensors to create, default is 0.
        LiDAR_Positions (list[tuple, tuple]): A list of tuples defining the positions and orientations (Quat) of each LiDAR sensor, 
                                if empty, random positions will be used.
        Camera_Positions (list[tuple, tuple]): A list of tuples defining the positions and orientations (Quat) of each camera sensor,
                                if empty, random positions will be used.
    """
    def __init__(self, num_lidars=5, num_cameras=5, lidar_positions=None, camera_positions=None, distance_lidar=3.0, distance_camera=2.5, include_bottom_lidar=True):
        self.Num_LiDARs = num_lidars
        self.Num_Cameras = num_cameras
        self.distance_lidar = distance_lidar
        self.distance_camera = distance_camera
        self.include_bottom_lidar = include_bottom_lidar

        if lidar_positions is not None:
            if len(lidar_positions) != num_lidars:
                raise ValueError("Length of lidar positions must match number of lidars.")
            elif not all(check_valid_quat(pos[1]) for pos in lidar_positions):
                raise ValueError("One or more LiDAR orientations are not valid quaternions.")
            else:
                self.LiDAR_Positions = lidar_positions
        else:
            if self.Num_LiDARs != 0:
                self.LiDAR_Positions = self.generate_random_position(
                    num_lidars,
                    distance_lidar=self.distance_lidar,
                    is_lidar=True,
                    include_bottom=self.include_bottom_lidar,
                )

        if camera_positions is not None:
            if len(camera_positions) != num_cameras:
                raise ValueError("Length of camera positions must match number of cameras.")
            elif not all(check_valid_quat(pos[1]) for pos in camera_positions):
                raise ValueError("One or more Camera orientations are not valid quaternions.")
            else:
                self.Camera_Positions = camera_positions
        else:
            if self.Num_Cameras != 0:
                self.Camera_Positions = self.generate_random_position(num_cameras, distance_camera=self.distance_camera, is_lidar=False)
            
    
    def build_sensors(self, stage, parent_path="/World/sensors"):
        """Build and create the sensors in the simulation environment.

        Args:
            distance (float): Distance from the origin to place the sensors if random positions are generated.
        
        Returns:
            lidar_render_products (list): A list of render products associated with the created LiDAR sensors.
            camera_render_products (list): A list of render products associated with the created camera sensors.
        """
        lidar_render_products = None, None
        camera_render_products = None
        
        if self.Num_LiDARs != 0:
            lidar_render_products = self.create_lidars(self.LiDAR_Positions, stage, parent_path=parent_path)
        if self.Num_Cameras != 0:
            camera_render_products = self.create_cameras(self.Camera_Positions, stage, parent_path=parent_path)

        return None



    def generate_random_position(self, num_sensors, distance_lidar=3.0, distance_camera=2.5, is_lidar=True, include_bottom=False):
        """Generate random positions and orientations for sensors around the origin.

        Note:
            Default orientation for LiDARs is + X, while for cameras it is - Z.

        Args:
            num_sensors (int): Number of sensors to generate positions for.
            distance (float): Distance from the origin to place the sensors.
            is_lidar (bool): Whether the sensors are LiDARs or cameras (affects orientation).
            
        Returns:
            sensor_configs (dict[tuple, tuple]): A list of tuples containing position and orientation (Quat) for each sensor.
        """

        sensor_configs = dict()
        if is_lidar:
            sensor_configs["top"] = ((0, 0, distance_lidar), quat_from_euler(0, 90, 0))
            if include_bottom:
                sensor_configs["bottom"] = ((0, 0, -distance_lidar), quat_from_euler(0, -90, 0))
        else:
            sensor_configs["top"] = ((0, 0, distance_camera), rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True))

        remaining = max(num_sensors - len(sensor_configs), 0)
        yaw_angles = np.linspace(0, 360, remaining, endpoint=False) if remaining > 0 else []

        # Determine the pitch angles randomly within specified ranges
        pitch_angles = np.random.uniform(30, 60, remaining) if remaining > 0 else []

        # Roll angles are fixed at 0 for all sensors Temporarily
        roll_angles = np.zeros(remaining)

        for yaw, pitch, roll in zip(yaw_angles, pitch_angles, roll_angles):
            if is_lidar:
                distance = distance_lidar
                orientation = quat_from_euler(yaw, pitch, roll)

            else:
                distance = distance_camera
                orientation = rot_utils.euler_angles_to_quats(np.array([roll, pitch, yaw+180]), degrees=True)
            x = distance * np.cos(np.radians(yaw)) * np.cos(np.radians(pitch))
            y = distance * np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
            z = distance * np.sin(np.radians(pitch))
            position = (x, y, z)

            sensor_configs[f"sensor_{len(sensor_configs)}"] = (position, orientation)
        return sensor_configs


    def create_lidars(self, configs, stage, parent_path="/World/sensors"):
        """Create LiDAR sensors in the simulation environment based on provided configurations.

        Args:
            configs (dict[tuple, tuple]): A list of tuples containing position and orientation (Quat) for each LiDAR sensor.

        Returns:
            render_products (list): A list of render products associated with the created LiDAR sensors.
        """

        lidars = []
        render_products = []
        annotators = []
        # stable_maps = []

        for name, (pos, quat) in configs.items():
            path = f"{parent_path}/{name}_Lidar"
            _, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path=path,
                parent=None,
                config="OS0_REV7_128ch10hz2048res",
                translation=pos,
                orientation=quat,
                visualize=False,
                force_camera_prim=False,
            )
            # TODO: Customize LiDAR parameters as needed

            sensor_prim = stage.GetPrimAtPath(path)
            sensor_prim.GetAttribute("visibility").Set(UsdGeom.Tokens.invisible)
            # Shrink minimum range to reduce near-field blind spots (keep high-density OS0 config).
            try:
                aux_attr = sensor_prim.GetAttribute("omni:sensor:Core:auxOutputType")
                if not aux_attr or not aux_attr.IsValid():
                    aux_attr = sensor_prim.CreateAttribute(
                        "omni:sensor:Core:auxOutputType",
                        Sdf.ValueTypeNames.Token,
                    )
                aux_attr.Set("EXTRA")

                min_attr = sensor_prim.GetAttribute("minRange")
                if not min_attr or not min_attr.IsValid():
                    min_attr = sensor_prim.CreateAttribute("minRange", Sdf.ValueTypeNames.Float)
                min_attr.Set(0.02)
                max_attr = sensor_prim.GetAttribute("maxRange")
                if not max_attr or not max_attr.IsValid():
                    max_attr = sensor_prim.CreateAttribute("maxRange", Sdf.ValueTypeNames.Float)
                max_attr.Set(2.0)
            except Exception as exc:
                print(f"[SensorBuilder][Warning] Failed to set min/max range for {path}: {exc}")

            # Render product resolution kept modest for performance.
            render_product = rep.create.render_product(sensor.GetPath(), resolution=(1024, 1024))
            annotator = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXLidarScanBuffer")
            annotator.initialize(
                outputDistance=True,
                outputAzimuth=True,
                outputIntensity=True,
                outputElevation=True,
                # outputMaterialId=True,
                # outputObjectId=True,
            )
            annotator.attach([render_product.path])
            # stable_map = rep.AnnotatorRegistry.get_annotator("StableIdMap")
            # stable_map.attach([render_product.path])
            render_products.append(render_product)
            lidars.append(sensor)
            annotators.append(annotator)
            print("[SensorBuilder] Created LiDAR sensor at:", path)
        
        # self.lidar_rp = list(zip(annotators, stable_maps, lidars))
        self.lidar_rp = list(zip(annotators, lidars))
        return None


    def create_cameras(self, configs, stage, parent_path="/World/sensors"):
        """Create camera sensors in the simulation environment based on provided configurations.

        Args:
            configs (dict[tuple, tuple]): A list of tuples containing position and orientation (Quat) for each camera sensor.
        
        Returns:
            render_products (list): A list of render products associated with the created camera sensors.
        """
        cameras = []
        extrinsics = []
        intrinsics = []
        for name, (pos, quat) in configs.items():
            path = f"{parent_path}/{name}_Camera"
            camera = Camera(
                prim_path=path,
                position=pos,
                orientation=quat,
                resolution=(800, 600),
                frequency=20,
            )
            prim = stage.GetPrimAtPath(path)
            prim.GetAttribute("focalLength").Set(24.0)
            prim.GetAttribute("horizontalAperture").Set(36.0)
            prim.GetAttribute("verticalAperture").Set(24.0)
            camera.set_clipping_range(0.001, 100.0)
            camera.initialize()
            # Record extrinsic/intrinsic for downstream export.
            
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
            extrinsics.append(np.array(xf, dtype=np.float32))
            focal = prim.GetAttribute("focalLength").Get()
            horiz_ap = prim.GetAttribute("horizontalAperture").Get()
            vert_ap = prim.GetAttribute("verticalAperture").Get()
            intr = np.array(
                [
                    [focal, 0, horiz_ap / 2.0],
                    [0, focal, vert_ap / 2.0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            intrinsics.append(intr)
            cameras.append(camera)
            print(f"[SensorBuilder] Created Isaac Camera at {path} with position: {pos}, orientation: {quat}")

        self.cameras = cameras
        self.camera_extrinsics = np.stack(extrinsics, axis=0) if extrinsics else np.zeros((0, 4, 4), dtype=np.float32)
        self.camera_intrinsics = np.stack(intrinsics, axis=0) if intrinsics else np.zeros((0, 3, 3), dtype=np.float32)
        return None


    def get_camera_outputs(self):
        """Get the RGBA outputs from all created cameras.

        Returns:
            images (list): A list of RGBA images captured by each camera.
        """
        images = []
        for camera in self.cameras:
            image = camera.get_rgba()[:, :, :3]
            images.append(image)

        return images

    def get_camera_transforms(self):
        """Return cached camera extrinsics and intrinsics."""
        return self.camera_extrinsics, self.camera_intrinsics

    
    def get_lidar_outputs(self, stage, return_metadata=False):
        """Get the LiDAR scan data from all created LiDAR sensors.

        Returns:
            scans (list): A list of LiDAR scan data from each sensor.
        """
        scan_points = []
        scan_intensities = []
        # material_ids = []
        # object_ids = []
        # stable_maps = []
        # for annotator, stable_map, sensor in self.lidar_rp:
        for annotator, sensor in self.lidar_rp:
            data = annotator.get_data()
            # print(data)
            if data and len(data["data"]) > 0:
                points_local = np.array(data["data"])
                if points_local.ndim == 1:
                    size = points_local.size
                    if size == 0:
                        continue
                    if size % 3 == 0:
                        points_local = points_local.reshape((-1, 3))
                    elif size % 4 == 0:
                        points_local = points_local.reshape((-1, 4))[:, :3]
                    else:
                        print(f"[SensorBuilder][Warning] Unexpected LiDAR data shape for {sensor.GetPath()}: {points_local.shape}")
                        continue
                if points_local.shape[1] < 3:
                    print(f"[SensorBuilder][Warning] LiDAR data has fewer than 3 cols for {sensor.GetPath()}: {points_local.shape}")
                    continue

                intensity = np.array(data["intensity"])
                if intensity.ndim == 0 or intensity.size == 0:
                    print(f"[SensorBuilder][Warning] Empty intensity for {sensor.GetPath()}, dropping this scan")
                    continue
                if intensity.ndim > 1:
                    intensity = intensity.reshape(-1)

                if intensity.size != points_local.shape[0]:
                    min_len = min(intensity.size, points_local.shape[0])
                    points_local = points_local[:min_len]
                    intensity = intensity[:min_len]

                if points_local.shape[0] == 0 or intensity.size == 0:
                    print(f"[SensorBuilder][Warning] Empty LiDAR return after trimming for {sensor.GetPath()}")
                    continue

                lidar_prim = stage.GetPrimAtPath(sensor.GetPath())
                points_world = transform_local_to_world(points_local, lidar_prim)
                scan_points.append(points_world)
                scan_intensities.append(intensity)
                # if return_metadata:
                #     material_ids.append(np.array(data.get("materialId", [])))
                #     object_ids.append(np.array(data.get("objectId", [])))
                #     stable_maps.append(stable_map.get_data())
            
        if len(scan_points) != 0 and len(scan_intensities) != 0:
            scan_points = np.concatenate(scan_points, axis=0)
            scan_intensities = np.concatenate(scan_intensities, axis=0)
        else:
            scan_points = np.zeros((0, 3), dtype=np.float32)
            scan_intensities = np.zeros((0,), dtype=np.float32)
        # if return_metadata:
        #     if len(material_ids):
        #         material_ids = np.concatenate(material_ids, axis=0)
        #     else:
        #         material_ids = np.zeros((0,), dtype=np.uint8)
        #     if len(object_ids):
        #         object_ids = np.concatenate(object_ids, axis=0)
        #     else:
        #         object_ids = np.zeros((0,), dtype=np.uint8)
        #     stable_maps = [m for m in stable_maps if m is not None]
        #     return scan_points, scan_intensities, material_ids, object_ids, stable_maps

        return scan_points, scan_intensities
