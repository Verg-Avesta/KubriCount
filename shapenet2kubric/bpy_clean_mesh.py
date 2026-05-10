# # Copyright 2024 The Kubric Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.


# import argparse
# from kubric.safeimport.bpy import bpy


# def cleanup_mesh(asset_id: str, source_path: str, target_path: str):
#   # start from a clean slate
#   bpy.ops.wm.read_factory_settings(use_empty=True)
#   bpy.context.scene.world = bpy.data.worlds.new("World")

#   # import source mesh
#   bpy.ops.import_scene.gltf(filepath=source_path, loglevel=50)

#   bpy.ops.object.select_all(action='DESELECT')

#   for obj in bpy.data.objects:
#     # remove duplicate vertices
#     bpy.context.view_layer.objects.active = obj
#     bpy.ops.object.mode_set(mode='EDIT')
#     bpy.ops.mesh.remove_doubles(threshold=1e-06)
#     bpy.ops.object.mode_set(mode='OBJECT')
#     # disable auto-smoothing
#     obj.data.use_auto_smooth = False
#     # split edges with an angle above 70 degrees (1.22 radians)
#     m = obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
#     m.split_angle = 1.22173
#     bpy.ops.object.modifier_apply(modifier="EdgeSplit")
#     # move every face an epsilon in the direction of its normal, to reduce clipping artifacts
#     m = obj.modifiers.new("Displace", "DISPLACE")
#     m.strength = 0.00001
#     bpy.ops.object.modifier_apply(modifier="Displace")

#   # join all objects together
#   bpy.ops.object.select_all(action='SELECT')
#   bpy.ops.object.join()

#   # set the name of the asset
#   bpy.context.active_object.name = asset_id

#   # export cleaned up mesh
#   bpy.ops.export_scene.gltf(filepath=str(target_path), check_existing=True)


# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument('--source_path', type=str)
#   parser.add_argument('--target_path', type=str)
#   parser.add_argument('--asset_id', type=str)
#   args = parser.parse_args()
#   cleanup_mesh(asset_id=args.asset_id,
#                source_path=args.source_path,
#                target_path=args.target_path)

import argparse
from kubric.safeimport.bpy import bpy


def cleanup_mesh(asset_id: str, source_path: str, target_path: str):
    # Clear the scene.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.world = bpy.data.worlds.new("World")

    # Import the GLB file.
    bpy.ops.import_scene.gltf(filepath=source_path, loglevel=50)
    
    print(f"Imported objects: {len(bpy.data.objects)}")
    for obj in bpy.data.objects:
        print(f"  - {obj.name} (type: {obj.type}, parent: {obj.parent.name if obj.parent else 'None'})")

    # ========== Step 1: detach all parent-child relationships while preserving world transforms ==========
    # CLEAR_KEEP_TRANSFORM must be used to preserve visual positions.
    
    # Sort by hierarchy depth and process the deepest child objects first.
    def get_depth(obj):
        depth = 0
        parent = obj.parent
        while parent:
            depth += 1
            parent = parent.parent
        return depth
    
    objects_by_depth = sorted(bpy.data.objects, key=get_depth, reverse=True)
    
    for obj in objects_by_depth:
        if obj.parent is not None:
            # Save the current world matrix.
            matrix_world_backup = obj.matrix_world.copy()
            
            # Select the object.
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            
            # Detach the parent-child relationship while preserving the transform.
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
            
            # Ensure the world matrix is correct.
            obj.matrix_world = matrix_world_backup

    # ========== Step 2: apply transforms to vertex data for each MESH object ==========
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    if not mesh_objects:
        print(f"Warning: No mesh objects found in {source_path}")
        bpy.ops.mesh.primitive_cube_add(size=0.001)
        obj = bpy.context.active_object
        obj.name = asset_id
        bpy.ops.export_scene.gltf(
            filepath=str(target_path),
            check_existing=True,
            use_selection=True
        )
        return

    print(f"Processing {len(mesh_objects)} mesh objects")

    for obj in mesh_objects:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        # Copy mesh data first if it is shared by multiple objects.
        bpy.ops.object.make_single_user(object=True, obdata=True, material=True)
        
        # Important: apply all transforms (location, rotation, scale) to vertex data.
        # This bakes the matrix_world transform into vertex coordinates.
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        print(f"  Applied transform to: {obj.name}")

    # ========== Step 3: delete all non-MESH objects ==========
    bpy.ops.object.select_all(action='DESELECT')
    for obj in list(bpy.data.objects):
        if obj.type != 'MESH':
            obj.select_set(True)
    
    if bpy.context.selected_objects:
        bpy.ops.object.delete()
        print(f"Deleted non-mesh objects")

    # ========== Step 4: join all MESH objects ==========
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    if not mesh_objects:
        print("Error: No mesh objects remaining after cleanup")
        return

    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = mesh_objects[0]
    
    if len(mesh_objects) > 1:
        bpy.ops.object.join()
        print(f"Joined {len(mesh_objects)} meshes")

    # ========== Step 5: clean the joined mesh ==========
    obj = bpy.context.active_object
    
    if obj and obj.type == 'MESH':
        # Remove duplicate vertices.
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=1e-06)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Disable auto-smoothing when supported by the Blender version.
        try:
            if hasattr(obj.data, 'use_auto_smooth'):
                obj.data.use_auto_smooth = False
        except:
            pass
        
        # EdgeSplit modifier.
        m = obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
        m.split_angle = 1.22173
        bpy.ops.object.modifier_apply(modifier="EdgeSplit")
        
        # Displace modifier.
        m = obj.modifiers.new("Displace", "DISPLACE")
        m.strength = 0.00001
        bpy.ops.object.modifier_apply(modifier="Displace")
        
        # Set the object name.
        obj.name = asset_id
        
        # Set the origin to the geometry center.
        # bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # ========== Step 6: final validation and export ==========
    remaining = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    if len(remaining) != 1:
        print(f"Warning: Expected 1 mesh, found {len(remaining)}")
        # Keep the first mesh and delete the rest.
        for extra in remaining[1:]:
            bpy.data.objects.remove(extra, do_unlink=True)
    
    # Select the final object.
    bpy.ops.object.select_all(action='DESELECT')
    final_obj = bpy.data.objects[0]
    final_obj.select_set(True)
    bpy.context.view_layer.objects.active = final_obj

    # Export.
    bpy.ops.export_scene.gltf(
        filepath=str(target_path),
        check_existing=True,
        use_selection=True,
        export_apply=False  # Transforms have already been applied.
    )
    
    print(f"Successfully exported to {target_path}")
    print(f"Final mesh: {len(final_obj.data.vertices)} vertices, {len(final_obj.data.polygons)} faces")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str)
    parser.add_argument('--target_path', type=str)
    parser.add_argument('--asset_id', type=str)
    args = parser.parse_args()
    cleanup_mesh(asset_id=args.asset_id,
                 source_path=args.source_path,
                 target_path=args.target_path)

