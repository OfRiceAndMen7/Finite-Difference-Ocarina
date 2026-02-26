import bpy
from mathutils import Vector

def align_to_first_octant():
    # Ensure we are in object mode
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    # Get the active object
    obj = bpy.context.active_object

    if not obj or obj.type != 'MESH':
        print("Please select a mesh object.")
        return

    # 1. Update the dependency graph to get the latest transforms
    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(dg)

    # 2. Find the min corner of the bounding box in world space
    # The bbox is local, so we multiply by the world matrix
    matrix_world = obj.matrix_world
    bbox_corners = [matrix_world @ Vector(corner) for corner in obj.bound_box]

    # Calculate the minimum x, y, and z values found in the corners
    min_x = min(c.x for c in bbox_corners)
    min_y = min(c.y for c in bbox_corners)
    min_z = min(c.z for c in bbox_corners)

    # 3. Calculate the translation vector to move min corner to (0,0,0)
    translation_vec = Vector((-min_x, -min_y, -min_z))

    # 4. Apply the translation to the object's location
    obj.location += translation_vec

    print(f"Aligned {obj.name} to origin. Offset applied: {translation_vec}")

# Run the function
align_to_first_octant()
