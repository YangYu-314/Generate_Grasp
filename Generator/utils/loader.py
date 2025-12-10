import json
import os

import omni.kit.commands
import omni
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, Sdf, UsdPhysics, Gf
from isaacsim.sensors.rtx import apply_nonvisual_material

from .utils import quat_from_euler, generate_default_setting, sanitize_name


DATASET_TO_ISAAC = {
    "Plastic": ("plastic", "paint", "none"),
    "Metal": ("steel", "clearcoat", "none"),
    "Ceramic": ("ceramic_glass", "clearcoat", "none"),
    "Stainless Metal": ("steel", "clearcoat", "retroreflective"),
    "Glass": ("clear_glass", "clearcoat", "single_sided"),
    "Stainless Steel": ("steel", "clearcoat", "retroreflective"),
    "Steel": ("steel", "clearcoat", "none"),

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
    "Aluminum": ("aluminum", "clearcoat", "retroreflective"),
    "Organic": ("leaf_grass", "none", "none"),
    "Foam with Leather Cover": ("leather", "paint_clearcoat", "none"),
    "Foam and Leatherette": ("leather", "paint_clearcoat", "none"),
    "Leather": ("leather", "paint_clearcoat", "none"),
    "Organic Material": ("leaf_grass", "none", "none"),
    "Water or Soil": ("mud", "none", "none"),
    "Plastic and Metal": ("plastic", "clearcoat", "none"),
}


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


class StageLoader:
    """Class for loading and managing stage, assets, and configurations.

    Attributes:
        assets_directory (str): The directory where assets are stored.
        assets_config_directory (str): The directory where asset configuration files are stored.
        is_table (bool): Flag indicating if the loader is for table environments.
    """

    def __init__(self, stage, assets_directory, assets_config_directory, material_directory, stage_config):
        self.stage = stage
        self.assets_directory = assets_directory
        self.assets_config_directory = assets_config_directory
        self.material_directory = material_directory
        self.stage_config = stage_config
        self.is_table = stage_config.get("mode", "simple") == "table"
    
    def load_asset_config(self, asset_name):
        config_path = os.path.join(self.assets_config_directory, f"{asset_name}.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        object_name = config["object_name"]
        category = config["category"]
        dimensions = config["dimension"]

        print("[AssetLoader] Loading asset config for:", object_name, "of category:", category)

        object_parts_info = []
        for part in config["parts"]:
            part_name = "_".join(part["name"].split(" "))
            part_name = sanitize_name(part_name)
            material = part["material"]
            label = part["label"]
            rank = part["priority_rank"]

            mesh_path = os.path.join(self.assets_directory, asset_name, f"objs/{label}.usd")
            material_path = os.path.join(self.material_directory, ISAAC_MATERIALS_MDL.get(material, ("Plastics", "Plastics/Plastic.mdl"))[1])
            material_name = ISAAC_MATERIALS_MDL.get(material, ("Plastic", "Plastics/Plastic.mdl"))[0]
            non_visual_tuple = DATASET_TO_ISAAC.get(material, ("plastic", "paint", "none"))
        
            print("[AssetLoader] Reading Config for part:", part_name, "with material:", material)
            
            part_info = {
                "part_name": part_name,
                "mesh_path": mesh_path,
                "material": material,
                "material_name": material_name,
                "material_path": material_path,
                "non_visual": non_visual_tuple,
                "rank": rank
            }
            object_parts_info.append(part_info)
        
        return object_parts_info
    

    def build_scene(self):
        """Construct the stage based on the provided configuration."""
        
        # Set up lighting
        omni.kit.commands.execute(
            "CreatePrim", prim_type="DistantLight",
            attributes={"inputs:angle": 1.0, "inputs:intensity": 3500}
        )

        # Create Looks scope
        if not self.stage.GetPrimAtPath("/Looks"):
            omni.kit.commands.execute(
                "CreatePrim", 
                prim_type="Scope", 
                prim_path="/Looks"
            )

        configs = self.stage_config.get("object_config", {})
        mode = configs.get("mode", "random")
        number_of_objects = configs.get("number", 1)

        # Read stage configuration
        assets_list = configs.get("objects", [])
        if mode == "random":
            assets_list = generate_default_setting(number_of_objects)

        # Set up environment for different stage types
        if self.is_table:
            print("[StageLoader] Building table environment.")
            self.spawn_table()
            for idx in range(len(assets_list)):
                self.spawn_asset(
                    asset_name=assets_list[idx]["name"],
                    asset_config=assets_list[idx],
                    root_path=f"/World/Assets/object_{idx}_obj{assets_list[idx]['name']}",
                    is_rigid=True
                )
        else:
            print("[StageLoader] Building static environment.")
            for idx in range(len(assets_list)):
                self.spawn_asset(
                    asset_name=assets_list[idx]["name"],
                    asset_config=assets_list[idx],
                    root_path=f"/World/Assets/object_{idx}_obj{assets_list[idx]['name']}",
                    is_rigid=False
                )
        print("[StageLoader] Finished building the scene.")


    def spawn_asset(self, asset_name, asset_config, root_path="/World/Assets/object", is_rigid=False):
        """Spawn an asset into the stage based on its configuration.
        
        Args:
            asset_name (str): The name of the asset to spawn.
            asset_config (dict): The configuration dictionary for the asset.
            root_path (str): The root path in the stage where the asset will be spawned, should in format "/World/Assets/asset_name".
            is_rigid (bool): Flag indicating if the asset should have rigid body physics.
        
        """
        object_parts_info = self.load_asset_config(asset_name)

        # Create root prim if it doesn't exist
        if not self.stage.GetPrimAtPath(root_path):
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Xform",
                prim_path=root_path
            )
        else:
            raise ValueError(f"Asset root path {root_path} already exists in the stage.")
        
        root_prim = self.stage.GetPrimAtPath(root_path)
        UsdPhysics.CollisionAPI.Apply(root_prim)

        if is_rigid:
            UsdPhysics.RigidBodyAPI.Apply(root_prim)
            collision_group_path = Sdf.Path(f"/World/CollisionGroups/{str(root_path).split('/')[-1]}")
            group = UsdPhysics.CollisionGroup.Define(self.stage, collision_group_path)
            parts_path = []

        for idx, part in enumerate(object_parts_info):
            part_name = part["part_name"]
            mesh_path = part["mesh_path"]
            material = part["material"]
            material_name = part["material_name"]
            material_path = part["material_path"]
            non_visual = part["non_visual"]

            prim_path = f"{root_path}/part_{idx}_{part_name}"
            omni.kit.commands.execute(
                "CreatePrim", 
                prim_path=prim_path, 
                prim_type="Xform"
            )

            omni.kit.commands.execute(
                "AddReferenceCommand",
                prim_path=prim_path,
                stage=self.stage,
                reference=Sdf.Reference(assetPath=mesh_path)
            )

            part_prim = self.stage.GetPrimAtPath(prim_path)
            print("[StageLoader] Spawned asset part at:", prim_path)
            
            # mtl_name 
            omni.kit.commands.execute(
                "CreateMdlMaterialPrim",
                mtl_url=material_path,
                mtl_name=material_name,
                mtl_path=f"/Looks/{part_prim.GetName()}_Mat",
            )
            
            mat_prim = self.stage.GetPrimAtPath(f"/Looks/{part_prim.GetName()}_Mat")
            apply_nonvisual_material(mat_prim, non_visual[0], non_visual[1], non_visual[2])

            omni.kit.commands.execute(
                "BindMaterial",
                prim_path=str(part_prim.GetPath()),
                material_path=str(mat_prim.GetPath())
            )
            print("[StageLoader] Applied material:", material, "to part:", part_name)

            UsdPhysics.CollisionAPI.Apply(part_prim)

            if is_rigid:
                parts_path.append(part_prim.GetPath())
        
        if is_rigid:
            collision_api = group.GetCollidersCollectionAPI()
            collision_api.GetIncludesRel().SetTargets(parts_path)
            group.CreateFilteredGroupsRel().AddTarget(collision_group_path)

        self.transform_asset(
            root_prim,
            translation=asset_config.get("position", (0, 0, 0)),
            rotation=asset_config.get("rotation", (0, 0, 0)),
            scale=asset_config.get("scale", (1, 1, 1))
        )


    def transform_asset(self, xform, translation=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        """Apply transformations to the spawned asset.
        
        Args:
            xform (Usd.Prim): The USD prim representing the asset to transform.
            translation (tuple): A tuple representing the translation (x, y, z).
            rotation (tuple): A tuple representing the rotation (yaw, pitch, roll) in degrees.
            scale (tuple): A tuple representing the scale (sx, sy, sz).
        """
        xform = UsdGeom.Xformable(xform)

        translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        translation_vec = Gf.Vec3f(*translation)
        if translate_ops:
            translate_ops[0].Set(translation_vec)
        else:
            xform.AddTranslateOp().Set(translation_vec)

        orient_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]
        quat = quat_from_euler(*rotation)
        if orient_ops:
            orient_ops[0].Set(quat)
        else:
            xform.AddOrientOp().Set(quat)

        scale_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
        scale_vec = Gf.Vec3f(*scale)
        if scale_ops:
            scale_ops[0].Set(scale_vec)
        else:
            xform.AddScaleOp().Set(scale_vec)

        print("[StageLoader] Transformed asset at:", xform.GetPath(), "with translation:", translation, "rotation:", rotation, "scale:", scale)


    def clear_stage(self):
        """Clear the current stage of all assets and configurations."""
        # Reset replicator state first so leftover render products do not hold stale annotators.
        try:
            rep.orchestrator.get_instance().reset()
        except Exception as exc:
            print(f"[StageLoader][Warning] Failed to reset replicator orchestrator: {exc}")

        world_prim = self.stage.GetPrimAtPath("/World")
        if world_prim.IsValid():
            for child in list(world_prim.GetChildren()):
                self.stage.RemovePrim(child.GetPath())
        else:
            self.stage.DefinePrim("/World", "Xform")

        # ===== 清空 /Looks 下的所有子 Prim =====
        looks_prim = self.stage.GetPrimAtPath("/Looks")
        if looks_prim.IsValid():
            for child in list(looks_prim.GetChildren()):
                self.stage.RemovePrim(child.GetPath())
        else:
            self.stage.DefinePrim("/Looks", "Scope")

        # Render products live outside /World; clear them to drop stale camera attachments.
        # render_prim = self.stage.GetPrimAtPath("/Render")
        # if render_prim.IsValid():
        #     for child in list(render_prim.GetChildren()):
        #         self.stage.RemovePrim(child.GetPath())
        # else:
        #     self.stage.DefinePrim("/Render", "Scope")

        print("[StageLoader] Cleared /World, /Looks, and /Render.")
        

    def spawn_table(self, path="/World/Table"):
        """Spawn a table into the stage if is_table is True."""
        # omni.kit.commands.execute(
        #     "AddGroundPlaneCommand",
        #     stage=self.stage,
        #     planePath=path,
        #     axis="Z",
        #     size=2000.0,
        #     position=(0.0, 0.0, 0.0),
        #     color=(0.5, 0.5, 0.5)
        # )

        if not self.stage.GetPrimAtPath(path):
            plane = UsdGeom.Mesh.Define(self.stage, Sdf.Path(path))
            half_size = 1000.0
            vertices = [
                (-half_size, -half_size, 0.0),
                ( half_size, -half_size, 0.0),
                ( half_size,  half_size, 0.0),
                (-half_size,  half_size, 0.0),
            ]
            face_vertex_counts = [4]
            face_vertex_indices = [0, 1, 2, 3]

            plane.CreatePointsAttr(vertices)
            plane.CreateFaceVertexCountsAttr(face_vertex_counts)
            plane.CreateFaceVertexIndicesAttr(face_vertex_indices)

            # 应用物理碰撞属性
            UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
