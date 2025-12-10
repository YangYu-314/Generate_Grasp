# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import sys

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

# Specify --/rtx-transient/stableIds/enabled=true to enable StableIdMap output
simulation_app = SimulationApp(
    {"headless": args.test, "enable_motion_bvh": True, "extra_args": ["--/rtx-transient/stableIds/enabled=true"]}
)

import carb
from isaacsim.sensors.rtx import LidarRtx, get_gmo_data
from isaacsim.storage.native import get_assets_root_path

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

import omni.timeline

# Load carter_warehouse_navigation scene
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

asset_path = assets_root_path + "/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/warehouse_navigation")

# Disable left camera render product to avoid interference with lidar
from isaacsim.core.utils.prims import get_prim_at_path

left_cam_rp_node = get_prim_at_path("/World/warehouse_navigation/Nova_Carter_ROS/front_hawk/left_camera_render_product")
left_cam_rp_node.GetAttribute("inputs:enabled").Set(False)

additional_lidar_attributes = {
    "omni:sensor:Core:auxOutputType": "FULL",
}

my_lidar = LidarRtx(
    prim_path="/World/warehouse_navigation/Nova_Carter_ROS/chassis_link/sensors/XT_32/PandarXT_32_10hz",
    name="lidar",
    **additional_lidar_attributes,
)
my_lidar.initialize()
my_lidar.attach_annotator("GenericModelOutput")
my_lidar.attach_annotator("StableIdMap")

i = 0
timeline = omni.timeline.get_timeline_interface()
timeline.play()
while simulation_app.is_running() and (not args.test or i < 10):
    simulation_app.update()

    # Convert the StableIdMap buffer to a dictionary of stable IDs to labels
    stable_id_map_buffer = my_lidar.get_current_frame()["StableIdMap"]
    stable_id_map = LidarRtx.decode_stable_id_mapping(stable_id_map_buffer.tobytes())

    # Get the object IDs from the GenericModelOutput buffer
    gmo_buffer = my_lidar.get_current_frame()["GenericModelOutput"]
    gmo_data = get_gmo_data(gmo_buffer)
    obj_ids = LidarRtx.get_object_ids(gmo_data.objId)

    # Print the object IDs and their labels
    for obj_id in set(obj_ids):
        if obj_id in stable_id_map:
            carb.log_warn(f"Object ID {obj_id} found in stable ID map: {stable_id_map[obj_id]}")
        else:
            carb.log_error(f"Object ID {obj_id} not found in stable ID map")

    i += 1

timeline.stop()
simulation_app.close()
