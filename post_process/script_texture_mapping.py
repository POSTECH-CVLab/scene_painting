import os
import glob
from time import time

import bpy
import bmesh

from mathutils import Vector
from bpy_extras.view3d_utils import location_3d_to_region_2d
# from bpy_extras.view3d_utils import region_2d_to_location_3d


def main(input_image_dir, frames_list=None, frames_step=1, object_list=None, use_subdivide_mesh=True, do_render=False,
         render_samples=4, texture_width=512, texture_height=512, save_mainfile=False, show_aggregated=True,
         debug=False):
    """
    :param input_image_dir: input image directory
    :param frames_list: frames list for texture mapping, when None get frames list from blend file
    :param frames_step: skips frames_step frames when getting frames list from scene, only used when frames_list is None
    :param object_list: objects used in texture mapping
    :param use_subdivide_mesh: subdivide_mesh before texture mapping
    :param do_render: render image for visualization
    :param texture_width: texture_width
    :param texture_height: texture_height
    """
    context = bpy.context

    # scene
    scene = context.scene
    scene.world = None  # set world data
    scene.cycles.max_bounces = 0  # set render setting for mask

    scene.render.tile_x = texture_width  # change tile size to speed up bake
    scene.render.tile_y = texture_height  # change tile size to speed up bake
    scene.cycles.samples = render_samples  # lower render sampling to speed up bake

    # create new collection
    collection = bpy.data.collections.get("UVProjection")
    if collection is None:
        collection = bpy.data.collections.new("UVProjection")
        scene.collection.children.link(collection)

    # set the frames
    if frames_list is None:
        frames_list = list(range(scene.frame_start, scene.frame_end + 1, frames_step))

        # get objects
    if object_list is None:
        object_list = [o for o in scene.objects if o.type == 'MESH']

    # initialize dict for bake
    frame_dict_list, object_dict_list = initialize_dict_list(input_image_dir, frames_list, object_list,
                                                             context, scene, collection,
                                                             texture_width=texture_width,
                                                             texture_height=texture_height,
                                                             )

    # subdivide mesh for bake
    if use_subdivide_mesh:
        subdivide_mesh(object_dict_list)

    mat_blank = create_mat_blank()

    # bake for each camera
    bake_scene(frame_dict_list, object_dict_list, context, scene, mat_blank)

    # aggregate
    aggregate_texture(frame_dict_list, object_dict_list)

    # render with aggregated texture
    if show_aggregated:
        visualize(context, object_dict_list, do_render=do_render)

    # save images to blend file
    save_images(frame_dict_list, object_dict_list, save_mainfile, debug=debug)

    if save_mainfile:
        bpy.ops.wm.save_mainfile()


def save_images(frame_dict_list, object_dict_list, save_mainfile, debug=False):
    # bpy.ops.file.unpack_all(method='WRITE_LOCAL')
    # bpy.ops.file.bpy.ops.file.pack_all()
    # for object_dict in object_dict_list:
    for object_dict in object_dict_list:
        aggregated_texture = object_dict['aggregated_texture']
        aggregated_texture.pack()

        baked_texture_list = object_dict['baked_texture_list']
        mask_occlusion_list = object_dict['mask_occlusion_list']
        mask_camera_list = object_dict['mask_camera_list']

        # -----
        for i, frame_dict in enumerate(frame_dict_list):
            baked_texture = baked_texture_list[i]
            mask_occlusion = mask_occlusion_list[i]
            mask_camera = mask_camera_list[i]
            if debug:
                baked_texture.pack()
                mask_occlusion.pack()
                mask_camera.pack()
            else:
                bpy.data.images.remove(baked_texture)
                bpy.data.images.remove(mask_occlusion)
                bpy.data.images.remove(mask_camera)

        mat_bake = object_dict['mat_bake']
        mat_mask_occlusion = object_dict['mat_mask_occlusion']
        mat_mask_camera = object_dict['mat_mask_camera']

        # -----
        if debug:
            pass
        else:
            obj = object_dict['object']
            obj.data.materials.clear()
            remove_material_from_object(obj, mat_bake)
            remove_material_from_object(obj, mat_mask_occlusion)
            remove_material_from_object(obj, mat_mask_camera)
            bpy.data.materials.remove(mat_bake)
            bpy.data.materials.remove(mat_mask_occlusion)
            bpy.data.materials.remove(mat_mask_camera)


def initialize_dict_list(input_image_dir, frames_list, object_list, context, scene, collection,
                         texture_width=512, texture_height=512):
    camera_list = create_cameras(frames_list, scene, collection)  # create cameras
    light_list = create_lights(frames_list, scene, collection)  # create lights
    input_image_list = load_images(input_image_dir)  # get images

    input_width, input_height = input_image_list[0].size  # get input image size
    white_img = create_white_image(width=input_width,
                                   height=input_height)  # create a white image for baking mask_camera

    frame_dict_list = [
        {'frame': frames_list[i],
         'camera': camera_list[i],
         'light': light_list[i],
         'input_image': input_image_list[frames_list[i] - 1]} for i in range(len(frames_list))]
    object_dict_list = [{'object': obj} for obj in object_list]
    # initialize objects
    for object_dict in object_dict_list:
        obj = object_dict['object']

        # images of frames per object
        baked_texture_list, mask_occlusion_list, mask_camera_list = create_images_for_bake(
            frames_list, obj, width=texture_width, height=texture_height)

        name = '{}.aggregated_texture'.format(obj.name)
        aggregated_texture = bpy.data.images.get(name)
        if aggregated_texture is not None:
            bpy.data.images.remove(aggregated_texture)

        aggregated_texture = bpy.data.images.new(name, width=512, height=512, alpha=True)
        aggregated_texture.use_fake_user = True

        # uv map
        projected_uvmap = create_projected_uvmap(obj, context)
        unwrapped_uvmap = create_unwrapped_uvmap(obj, context)

        # materials
        mat_bake = create_mat_bake("{}.bake".format(obj.name), projected_uvmap, unwrapped_uvmap)
        mat_mask_occlusion = create_mat_mask_occlusion("{}.mask_occlusion".format(obj.name), unwrapped_uvmap)
        mat_mask_camera = create_mat_mask_camera("{}.mask_camera".format(obj.name), projected_uvmap, unwrapped_uvmap,
                                                 white_image=white_img)
        mat_aggregated = create_mat_aggregated("{}.aggregated".format(obj.name), unwrapped_uvmap, aggregated_texture)

        object_dict['baked_texture_list'] = baked_texture_list
        object_dict['mask_occlusion_list'] = mask_occlusion_list
        object_dict['mask_camera_list'] = mask_camera_list
        object_dict['mat_bake'] = mat_bake
        object_dict['mat_mask_occlusion'] = mat_mask_occlusion
        object_dict['mat_mask_camera'] = mat_mask_camera
        object_dict['mat_aggregated'] = mat_aggregated
        object_dict['projected_uvmap'] = projected_uvmap
        object_dict['unwrapped_uvmap'] = unwrapped_uvmap
        object_dict['aggregated_texture'] = aggregated_texture

    return frame_dict_list, object_dict_list


def subdivide_mesh(object_dict_list):
    target_length = 0.45

    for object_dict in object_dict_list:
        obj = object_dict['object']

        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)

        while True:

            # select faces
            # FACTOR = 6
            # for f in bm.faces:
            #     f.select = f.calc_area() > FACTOR * avg_face_area
            #     f.select = f.calc_area() > FACTOR

            # selet edges
            for e in bm.edges:
                e.select = e.calc_length() > target_length
            edges = [edge for edge in bm.edges if edge.select]

            if len(edges) == 0:
                break

            bmesh.ops.subdivide_edges(bm, edges=edges, cuts=1, use_grid_fill=True)

        bm.to_mesh(mesh)
        bm.free()

    print('mesh subdivision complete')
    return


def bake_scene(frame_dict_list, object_dict_list, context, scene, mat_blank):
    camera_scene = scene.camera

    for i, frame_dict in enumerate(frame_dict_list):

        # ----- set camera, light, material, uvmap -----
        light = frame_dict['light']
        scene.camera = frame_dict['camera']  # set camera

        for object_dict in object_dict_list:
            set_projected_uvmap(object_dict['projected_uvmap'], object_dict['object'], scene.camera, scene)
            set_material_image_for_bake(object_dict['mat_bake'], frame_dict['input_image'],
                                        object_dict['baked_texture_list'][i],
                                        object_dict['mat_mask_occlusion'], object_dict['mask_occlusion_list'][i],
                                        object_dict['mat_mask_camera'], object_dict['mask_camera_list'][i])

        context.view_layer.objects.active = None
        for obj in bpy.data.objects:
            obj.select_set(False)

        # use flat shading
        for object_dict in object_dict_list:
            obj = object_dict['object']
            obj.data.use_auto_smooth = False
            for f in obj.data.polygons:
                f.use_smooth = False

        for object_dict in object_dict_list:
            obj = object_dict['object']

            # check visible
            flag = select_camera_border(frame_dict['camera'], context)
            selected = [obj.name for obj in bpy.context.selected_objects]
            bpy.ops.object.select_all(action='DESELECT')
            # print(selected)
            if flag and obj.name not in selected:
                # print('skip')
                continue

            # ----- bake mask_occlusion -----
            light.hide_render = False  # unhide light
            material = object_dict['mat_mask_occlusion']

            context.view_layer.objects.active = obj  # set active object
            obj.select_set(True)  # select object
            bpy.ops.object.mode_set(mode='EDIT')  # entering edit mode
            bpy.ops.mesh.select_all(action='SELECT')  # select all objects elements
            assign_material_to_object(obj, material)  # assign material to object

            # set active node
            nodes = material.node_tree.nodes
            texture_node = nodes.get('mask_occlusion')
            texture_node.select = True
            nodes.active = texture_node

            bpy.ops.object.bake(type='DIFFUSE', uv_layer=object_dict['unwrapped_uvmap'])

            remove_material_from_object(obj, material)  # remove material from object
            # bpy.ops.mesh.select_all(action='DESELECT')  # deselect all objects elements
            # bpy.ops.object.mode_set(mode='OBJECT')  # exiting edit mode
            # context.view_layer.objects.active = None
            # obj.select_set(False)
            light.hide_render = True  # hide light

            # ----- bake mask_camera -----
            material = object_dict['mat_mask_camera']

            context.view_layer.objects.active = obj  # set active object
            # obj.select_set(True)  # select object
            # bpy.ops.object.mode_set(mode='EDIT')  # entering edit mode
            # bpy.ops.mesh.select_all(action='SELECT')  # select all objects elements
            assign_material_to_object(obj, material)  # assign material to object

            # set active node
            nodes = material.node_tree.nodes
            texture_node = nodes.get('mask_camera')
            texture_node.select = True
            nodes.active = texture_node

            bpy.ops.object.bake(type='EMIT', uv_layer=object_dict['unwrapped_uvmap'])

            remove_material_from_object(obj, material)  # remove material from object
            # bpy.ops.mesh.select_all(action='DESELECT')  # deselect all objects elements
            # bpy.ops.object.mode_set(mode='OBJECT')  # exiting edit mode
            # context.view_layer.objects.active = None
            # obj.select_set(False)

            # ----- bake texture -----
            material = object_dict['mat_bake']

            # context.view_layer.objects.active = obj  # set active object
            # obj.select_set(True)  # select object
            # bpy.ops.object.mode_set(mode='EDIT')  # entering edit mode
            # bpy.ops.mesh.select_all(action='SELECT')  # select all objects elements
            assign_material_to_object(obj, material)  # assign material to object

            # set active uvmap
            # idx = obj.data.uv_layers.find(object_dict['unwrapped_uvmap'])
            # obj.data.uv_layers.active_index = idx

            # set active node
            nodes = material.node_tree.nodes
            # for n in nodes:
            #     n.select = False
            texture_node = nodes.get('baked_texture')
            texture_node.select = True
            nodes.active = texture_node

            bpy.ops.object.bake(type='EMIT', uv_layer=object_dict['unwrapped_uvmap'])

            remove_material_from_object(obj, material)  # remove material from object

            assign_material_to_object(obj, mat_blank)

            bpy.ops.mesh.select_all(action='DESELECT')  # deselect all objects elements
            bpy.ops.object.mode_set(mode='OBJECT')  # exiting edit mode
            context.view_layer.objects.active = None
            obj.select_set(False)

        print('baked frame {}'.format(frame_dict['frame']))

    context.view_layer.objects.active = None
    for obj in bpy.data.objects:
        obj.select_set(False)

    scene.camera = camera_scene

    print('bake complete')


def aggregate_texture(frame_dict_list, object_dict_list):
    for object_dict in object_dict_list:
        baked_texture_list = object_dict['baked_texture_list']
        mask_occlusion_list = object_dict['mask_occlusion_list']
        mask_camera_list = object_dict['mask_camera_list']
        aggregated_texture = object_dict['aggregated_texture']

        mask_sum_vec = None
        aggreated_texture_vec = None
        for i, frame_dict in enumerate(frame_dict_list):
            baked_texture = baked_texture_list[i]
            mask_occlusion = mask_occlusion_list[i]
            mask_camera = mask_camera_list[i]

            # get mask
            baked_texture_vec = Vector(baked_texture.pixels)
            mask_occlusion_vec = Vector(mask_occlusion.pixels)
            mask_camera_vec = Vector(mask_camera.pixels)

            mask_vec = mask_occlusion_vec * mask_camera_vec  # multiply masks to get true mask
            baked_texture_vec = baked_texture_vec * mask_vec  # masking occluded region in baked texture
            del mask_camera_vec
            del mask_occlusion_vec

            # weighted average
            if aggreated_texture_vec is None:
                aggreated_texture_vec = Vector.Fill(len(baked_texture_vec))
                mask_sum_vec = Vector.Fill(len(baked_texture_vec))
            aggreated_texture_vec = aggreated_texture_vec + baked_texture_vec
            del baked_texture_vec
            mask_sum_vec = mask_sum_vec + mask_vec
            del mask_vec

        for i, x in enumerate(mask_sum_vec):
            if x != 0:
                mask_sum_vec[i] = 1 / x
        aggreated_texture_vec = aggreated_texture_vec * mask_sum_vec
        del mask_sum_vec
        aggregated_texture.pixels = aggreated_texture_vec.to_tuple()
        del aggreated_texture_vec

    print('aggregation complete')


def visualize(context, object_dict_list, do_render=False):
    context.view_layer.objects.active = None
    for obj in bpy.data.objects:
        obj.select_set(False)

    # assign material
    for object_dict in object_dict_list:
        obj = object_dict['object']
        material = object_dict['mat_aggregated']
        # material = object_dict['mat_bake']  # fixme for debug
        # material = object_dict['mat_mask_occlusion'] # fixme for debug
        # material = object_dict['mat_mask_camera'] # fixme for debug

        context.view_layer.objects.active = obj  # set active object
        obj.select_set(True)  # select object
        bpy.ops.object.mode_set(mode='EDIT')  # entering edit mode
        bpy.ops.mesh.select_all(action='SELECT')  # select all objects elements
        assign_material_to_object(obj, material)  # assign material to object

        # remove_material_from_object(obj, material)  # remove material from object
        bpy.ops.mesh.select_all(action='DESELECT')  # deselect all objects elements
        bpy.ops.object.mode_set(mode='OBJECT')  # exiting edit mode
        context.view_layer.objects.active = None
        obj.select_set(False)
    print('assigned material to objects')

    # render
    if do_render:
        bpy.ops.render.render(animation=True)
        print('render complete')


def assign_material_to_object(obj, material):
    if material.name not in obj.data.materials:
        obj.data.materials.append(material)
    idx = obj.data.materials.find(material.name)
    obj.active_material_index = idx
    bpy.ops.object.material_slot_assign()

    # print(idx)
    # for p in obj.data.polygons[:]:
    #     print(p.material_index)
    #     p.material_index = idx


#
# def assign_material_to_object2(obj, material):
#     if obj.data.materials:
#         # assign to 1st material slot
#         obj.data.materials[0] = material
#     else:
#         # no slots
#         obj.data.materials.append(material)


def remove_material_from_object(obj, material):
    idx = obj.data.materials.find(material.name)
    if idx != -1:
        obj.data.materials.pop(index=idx)


def set_material_image_for_bake(mat_bake, input_image, baked_texture,
                                mat_mask_occlusion, mask_occlusion,
                                mat_mask_camera, mask_camera):
    # ----- mat_bake -----
    nodes = mat_bake.node_tree.nodes

    # input_image node
    input_image_node = nodes.get('input_image')
    input_image_node.image = input_image

    # baked_texture node
    baked_texture_node = nodes.get('baked_texture')
    baked_texture_node.image = baked_texture

    # ----- mat_mask_occlusion -----
    nodes = mat_mask_occlusion.node_tree.nodes

    # mask_occlusion node
    mask_occlusion_node = nodes.get('mask_occlusion')
    mask_occlusion_node.image = mask_occlusion

    # ----- mat_mask_camera -----
    nodes = mat_mask_camera.node_tree.nodes

    # mask_camera node
    mask_camera_node = nodes.get('mask_camera')
    mask_camera_node.image = mask_camera

    return


def set_projected_uvmap(uvmap, obj, camera, scene):
    # override = context.copy()
    # override['selected_objects'] = list(context.scene.objects)
    # override = {}

    # old_obj = context.view_layer.objects.active
    # context.view_layer.objects.active = obj  # set active object
    # bpy.ops.object.mode_set(mode='EDIT')  # entering edit mode

    # bpy.ops.mesh.select_all(action='SELECT')  # select all objects elements

    # for area in context.screen.areas:
    #     if area.type == 'VIEW_3D':
    #         for region in area.regions:
    #             if region.type == 'WINDOW':
    #                 # set active uvmap
    #                 idx = obj.data.uv_layers.find(uvmap)
    #                 obj.data.uv_layers.active_index = idx
    #
    #                 # set camera
    #                 override = bpy.context.copy()
    #                 override['area'] = area
    #                 # bpy.ops.view3d.view_camera(override)
    #                 region = area.spaces[0].region_3d
    #                 if region:
    #                     region.view_perspective = 'CAMERA'
    #                     bpy.ops.view3d.view_camera(override)
    #                     bpy.ops.view3d.view_camera(override)
    #                     # region.view_perspective = 'CAMERA'
    #
    #                 # the actual unwrapping operation
    #                 override = bpy.context.copy()
    #                 override['area'] = area
    #                 override['region'] = region
    #                 override['edit_object'] = obj
    #                 bpy.ops.uv.project_from_view(override)
    #         break

    # bpy.ops.object.mode_set(mode='OBJECT')  # exiting edit mode
    # context.view_layer.objects.active = old_obj

    toCameraMatrix = camera.matrix_world.inverted() @ obj.matrix_world
    # The frame is composed of the coordinates in the camera view
    frame = [v / v.z for v in camera.data.view_frame(scene=scene)]
    # Get the X, Y corners
    minX = min(v.x for v in frame)
    maxX = max(v.x for v in frame)
    minY = min(v.y for v in frame)
    maxY = max(v.y for v in frame)
    # Precalculations to avoid to repeat them when applied to the model
    deltaX = maxX - minX
    deltaY = maxY - minY
    offsetX = minX / deltaX
    offsetY = minY / deltaY

    clip_to_bounds = False

    def calc_uv(obj_co):

        # Object in camera view
        camCo = toCameraMatrix @ obj_co
        # Z is "inverted" as camera view is pointing to -Z of the camera
        z = -camCo.z
        try:
            # Translates x and y to UV coordinates
            x = (camCo.x / (deltaX * z)) - offsetX
            y = (camCo.y / (deltaY * z)) - offsetY

            if clip_to_bounds:
                minx = minX / deltaX - offsetX
                maxx = maxX / deltaX - offsetX
                miny = minY / deltaY - offsetY
                maxy = maxY / deltaY - offsetY
                # print(minx, x, maxx)
                x = min(max(minx, x), maxx)
                y = min(max(miny, y), maxy)

            return x, y, z
        except:
            # In case Z is zero
            return None

    uvmap = obj.data.uv_layers[uvmap]
    loops = obj.data.loops
    vertices = obj.data.vertices

    projected_co = {}  # Storage to avoid multiple calculations of the same world_to_camera_view
    # Go through all polygons
    for p in obj.data.polygons:
        # Calculate each vertex uv projection
        for i, vi in [(i, loops[i].vertex_index) for i in p.loop_indices]:
            if vi not in projected_co:  # not already calculated for this cam
                xyz = calc_uv(vertices[vi].co)
                if xyz is not None:
                    projected_co[vi] = (xyz[0], xyz[1])
                else:
                    projected_co[vi] = None
            if projected_co[vi] is not None:
                uvmap.data[i].uv = projected_co[vi]


def create_white_image(width=256, height=256):
    name = 'white_image'
    white_img = bpy.data.images.get(name)
    if white_img is not None:
        bpy.data.images.remove(white_img)

    white_img = bpy.data.images.new(name, width=width, height=height, alpha=False)
    # width = white_img.size[0]
    # height = white_img.size[1]
    # channels = white_img.channels
    pixels = [1.0 for i in range(len(white_img.pixels))]
    white_img.pixels = pixels
    return white_img


def create_mat_bake(name, projected_uvmap, unwrapped_uvmap):
    material = bpy.data.materials.get(name)
    if material is not None:
        bpy.data.materials.remove(material)
    material = bpy.data.materials.new(name=name)

    material.use_nodes = True
    material.node_tree.nodes.clear()
    material.use_fake_user = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    material_output = nodes.new("ShaderNodeOutputMaterial")
    material_output.name = 'Material Output'

    # ----- render -----

    # projected_uvmap node
    projected_uvmap_node = nodes.new("ShaderNodeUVMap")
    projected_uvmap_node.uv_map = projected_uvmap

    # input_image node
    input_image_node = nodes.new('ShaderNodeTexImage')
    input_image_node.name = 'input_image'
    # input_image_node.image = input_image
    input_image_node.extension = 'CLIP'
    links.new(input_image_node.inputs['Vector'], projected_uvmap_node.outputs['UV'])

    # emission node
    emission_node = nodes.new("ShaderNodeEmission")
    links.new(emission_node.inputs['Color'], input_image_node.outputs['Color'])
    links.new(material_output.inputs['Surface'], emission_node.outputs['Emission'])

    # # light path node
    # light_path_node = nodes.new('ShaderNodeLightPath')
    #
    # # Mix Shader node
    # mix_node = nodes.new('ShaderNodeMixShader')
    # links.new(mix_node.inputs['Fac'], light_path_node.outputs['Is Camera Ray'])
    # links.new(mix_node.inputs[2], emission_node.outputs['Emission'])
    # links.new(material_output.inputs['Surface'], mix_node.outputs['Shader'])

    # ----- bake -----
    # unwrapped_uvmap node
    unwrapped_uvmap_node = nodes.new("ShaderNodeUVMap")
    unwrapped_uvmap_node.uv_map = unwrapped_uvmap

    # baked_texture node
    baked_texture_node = nodes.new('ShaderNodeTexImage')
    baked_texture_node.name = 'baked_texture'
    # baked_texture_node.image = baked_texture
    baked_texture_node.extension = 'CLIP'
    links.new(baked_texture_node.inputs['Vector'], unwrapped_uvmap_node.outputs['UV'])

    return material


def create_mat_mask_occlusion(name, unwrapped_uvmap):
    material = bpy.data.materials.get(name)
    if material is not None:
        bpy.data.materials.remove(material)
    material = bpy.data.materials.new(name=name)

    material.use_nodes = True
    material.node_tree.nodes.clear()
    material.use_fake_user = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    material_output = nodes.new("ShaderNodeOutputMaterial")
    material_output.name = 'Material Output'

    # ----- render -----

    # diffuse node
    # diffuse_node = nodes.new("ShaderNodeBsdfDiffuse")
    # diffuse_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    # diffuse_node.inputs['Roughness'].default_value = 1.0

    diffuse_node = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    diffuse_node.inputs['Roughness'].default_value = 1.0

    links.new(material_output.inputs['Surface'], diffuse_node.outputs['BSDF'])

    # ----- bake -----
    # unwrapped_uvmap node
    unwrapped_uvmap_node = nodes.new("ShaderNodeUVMap")
    unwrapped_uvmap_node.uv_map = unwrapped_uvmap

    # baked_mask node
    mask_occlusion_node = nodes.new('ShaderNodeTexImage')
    mask_occlusion_node.name = 'mask_occlusion'
    # mask_occlusion_node.image = ...
    mask_occlusion_node.extension = 'CLIP'
    links.new(mask_occlusion_node.inputs['Vector'], unwrapped_uvmap_node.outputs['UV'])

    return material


def create_mat_mask_camera(name, projected_uvmap, unwrapped_uvmap, white_image):
    material = bpy.data.materials.get(name)
    if material is not None:
        bpy.data.materials.remove(material)
    material = bpy.data.materials.new(name=name)

    material.use_nodes = True
    material.node_tree.nodes.clear()

    material.use_fake_user = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    material_output = nodes.new("ShaderNodeOutputMaterial")
    material_output.name = 'Material Output'

    # ----- render -----

    # projected_uvmap node
    projected_uvmap_node = nodes.new("ShaderNodeUVMap")
    projected_uvmap_node.uv_map = projected_uvmap

    # white_image node
    white_image_node = nodes.new('ShaderNodeTexImage')
    white_image_node.image = white_image
    white_image_node.extension = 'CLIP'
    links.new(white_image_node.inputs['Vector'], projected_uvmap_node.outputs['UV'])

    # emission node
    emission_node = nodes.new("ShaderNodeEmission")
    links.new(emission_node.inputs['Color'], white_image_node.outputs['Color'])
    links.new(material_output.inputs['Surface'], emission_node.outputs['Emission'])

    # # light path node
    # light_path_node = nodes.new('ShaderNodeLightPath')
    #
    # # Mix Shader node
    # mix_node = nodes.new('ShaderNodeMixShader')
    # links.new(mix_node.inputs['Fac'], light_path_node.outputs['Is Camera Ray'])
    # links.new(mix_node.inputs[2], emission_node.outputs['Emission'])
    # links.new(material_output.inputs['Surface'], mix_node.outputs['Shader'])

    # ----- bake -----
    # unwrapped_uvmap node
    unwrapped_uvmap_node = nodes.new("ShaderNodeUVMap")
    unwrapped_uvmap_node.uv_map = unwrapped_uvmap

    # mask_camera_node
    mask_camera_node = nodes.new('ShaderNodeTexImage')
    mask_camera_node.name = 'mask_camera'
    # mask_camera_node.image = ...
    mask_camera_node.extension = 'CLIP'
    links.new(mask_camera_node.inputs['Vector'], unwrapped_uvmap_node.outputs['UV'])

    return material


def create_mat_aggregated(name, unwrapped_uvmap, aggregated_image):
    material = bpy.data.materials.get(name)
    if material is not None:
        bpy.data.materials.remove(material)
    material = bpy.data.materials.new(name=name)

    material.use_nodes = True
    material.node_tree.nodes.clear()
    material.use_fake_user = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    material_output = nodes.new("ShaderNodeOutputMaterial")
    material_output.name = 'Material Output'

    # ----- render -----

    # unwrapped_uvmap node
    unwrapped_uvmap_node = material.node_tree.nodes.new("ShaderNodeUVMap")
    unwrapped_uvmap_node.uv_map = unwrapped_uvmap

    # aggregated_image node
    aggregated_image_node = material.node_tree.nodes.new('ShaderNodeTexImage')
    aggregated_image_node.extension = 'CLIP'
    aggregated_image_node.name = 'aggregated_image'
    aggregated_image_node.image = aggregated_image
    links.new(aggregated_image_node.inputs['Vector'], unwrapped_uvmap_node.outputs['UV'])

    # emission node
    emission_node = material.node_tree.nodes.new("ShaderNodeEmission")
    links.new(emission_node.inputs['Color'], aggregated_image_node.outputs['Color'])
    links.new(material_output.inputs['Surface'], emission_node.outputs['Emission'])

    return material

def create_mat_aggregated_bsdf(name, unwrapped_uvmap, aggregated_image):
    material = bpy.data.materials.get(name)
    if material is not None:
        bpy.data.materials.remove(material)
    material = bpy.data.materials.new(name=name)

    material.use_nodes = True
    material.node_tree.nodes.clear()
    material.use_fake_user = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    material_output = nodes.new("ShaderNodeOutputMaterial")
    material_output.name = 'Material Output'

    # ----- render -----

    # unwrapped_uvmap node
    unwrapped_uvmap_node = material.node_tree.nodes.new("ShaderNodeUVMap")
    unwrapped_uvmap_node.uv_map = unwrapped_uvmap

    # aggregated_image node
    aggregated_image_node = material.node_tree.nodes.new('ShaderNodeTexImage')
    aggregated_image_node.extension = 'CLIP'
    aggregated_image_node.name = 'aggregated_image'
    aggregated_image_node.image = aggregated_image
    links.new(aggregated_image_node.inputs['Vector'], unwrapped_uvmap_node.outputs['UV'])

    # bsdf node
    bsdf_node = material.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf_node.inputs['Base Color'], aggregated_image_node.outputs['Color'])
    links.new(material_output.inputs['Surface'], bsdf_node.outputs['BSDF'])

    return material

def create_mat_blank(name='blank'):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)

    material.use_nodes = True
    material.node_tree.nodes.clear()
    material.use_fake_user = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    material_output = nodes.new("ShaderNodeOutputMaterial")
    material_output.name = 'Material Output'

    return material


def create_projected_uvmap(obj, context):
    name = 'projected_uvmap.{}'.format(obj.name)

    uvmap = obj.data.uv_layers.get(name, None)
    if uvmap is None:
        uvmap = obj.data.uv_layers.new(name=name)

    return name


def create_unwrapped_uvmap(obj, context):
    name = 'unwrapped_uvmap.{}'.format(obj.name)

    context.view_layer.objects.active = None
    for o in bpy.data.objects:
        o.select_set(False)

    context.view_layer.objects.active = obj
    obj.select_set(True)

    uvmap = obj.data.uv_layers.get(name, None)
    if uvmap is None:
        uvmap = obj.data.uv_layers.new(name=name)

    # override = context.copy()

    uvmap.active = True  # set active uvmap
    old_obj = context.view_layer.objects.active
    context.view_layer.objects.active = obj  # set active object
    bpy.ops.object.mode_set(mode='EDIT')  # entering edit mode

    bpy.ops.mesh.select_all(action='SELECT')  # select all objects elements
    bpy.ops.uv.smart_project()  # the actual unwrapping operation

    bpy.ops.object.mode_set(mode='OBJECT')  # exiting edit mode
    context.view_layer.objects.active = old_obj

    context.view_layer.objects.active = None

    return name


def create_images_for_bake(frames_list, obj, width=512, height=512):
    baked_texture_list = []
    mask_occlusion_list = []
    mask_camera_list = []
    for f in frames_list:
        name = '{}.baked_texture.{}'.format(obj.name, f)
        baked_texture = bpy.data.images.get(name)
        if baked_texture is not None:
            bpy.data.images.remove(baked_texture)
        baked_texture = bpy.data.images.new(name, width=width, height=height, alpha=False)
        baked_texture.use_fake_user = True

        name = '{}.mask_occlusion.{}'.format(obj.name, f)
        mask_occlusion = bpy.data.images.get(name)
        if mask_occlusion is not None:
            bpy.data.images.remove(mask_occlusion)
        mask_occlusion = bpy.data.images.new(name, width=width, height=height, alpha=False)
        mask_occlusion.use_fake_user = True

        name = '{}.mask_camera.{}'.format(obj.name, f)
        mask_camera = bpy.data.images.get(name)
        if mask_camera is not None:
            bpy.data.images.remove(mask_camera)
        mask_camera = bpy.data.images.new(name, width=width, height=height, alpha=False)
        mask_camera.use_fake_user = True

        baked_texture_list.append(baked_texture)
        mask_occlusion_list.append(mask_occlusion)
        mask_camera_list.append(mask_camera)
    return baked_texture_list, mask_occlusion_list, mask_camera_list


def load_images(input_image_dir):
    input_image_paths = sorted(glob.glob(os.path.join(input_image_dir, '*')))
    input_images = []
    for p in input_image_paths:
        img = bpy.data.images.load(p, check_existing=True)
        input_images.append(img)

    return input_images


def create_lights(frames_list, scene, collection, energy=10000000000):
    camera_scene = scene.camera
    lights_list = []
    for f in frames_list:
        scene.frame_set(f)

        name = "light_data.{}".format(f)
        light_data = bpy.data.lights.get(name)
        if light_data is None:
            light_data = bpy.data.lights.new(name=name, type='SPOT')

        light_data.use_nodes = True

        nodes = light_data.node_tree.nodes
        links = light_data.node_tree.links

        emission_node = nodes.get("Emission")
        light_falloff_node = nodes.new("ShaderNodeLightFalloff")
        links.new(emission_node.inputs['Strength'], light_falloff_node.outputs['Constant'])

        light_falloff_node.inputs['Strength'].default_value = energy
        light_data.falloff_type = 'CONSTANT'
        light_data.spot_size = 3.14159
        light_data.spot_blend = 0
        light_data.shadow_soft_size = 0
        light_data.cycles.max_bounces = 0

        name = "light.{}".format(f)
        light_object = bpy.data.objects.get(name)
        if light_object is None:
            light_object = bpy.data.objects.new(name=name, object_data=light_data)
            collection.objects.link(light_object)

        light_object.location = camera_scene.matrix_world.translation
        light_object.rotation_euler = camera_scene.matrix_world.to_euler()

        light_object.hide_viewport = True
        light_object.hide_render = True

        lights_list.append(light_object)

    return lights_list


def create_cameras(frames_list, scene, collection):
    camera_scene = scene.camera
    camera_list = []
    for f in frames_list:
        scene.frame_set(f)

        name = 'camera.{}'.format(f)
        camera_proj = bpy.data.objects.get(name)
        if camera_proj is not None:
            bpy.data.objects.remove(camera_proj)

        camera_proj = camera_scene.copy()
        camera_proj.name = name

        camera_proj.parent = None
        camera_proj.animation_data_clear()
        camera_proj.matrix_local = camera_proj.matrix_world

        camera_proj.hide_viewport = True

        camera_list.append(camera_proj)
        collection.objects.link(camera_proj)

    return camera_list


def view3d_find():
    # returns first 3d view, normally we get from context
    for area in bpy.context.window.screen.areas:
        if area.type == 'VIEW_3D':
            v3d = area.spaces[0]
            rv3d = v3d.region_3d
            for region in area.regions:
                if region.type == 'WINDOW':
                    return area, region, rv3d
    return None, None, None


def view3d_camera_border(scene, area_region_rv3d=None):
    obj = scene.camera
    cam = obj.data

    frame_local = cam.view_frame(scene=scene)

    # move from object-space into world-space
    frame_world = [obj.matrix_world @ v for v in frame_local]

    # move into pixelspace
    if area_region_rv3d is None:
        area_region_rv3d = view3d_find()
    area, region, rv3d = area_region_rv3d

    frame_px = [location_3d_to_region_2d(region, rv3d, v) for v in frame_world]
    # print("Camera frame 2d:", frame_px)

    # frame_debug = [region_2d_to_location_3d(region, rv3d, v, Vector((0.0, 0.0, 0.0))) for v in frame_px]
    # frame_debug = [region_2d_to_location_3d(region, rv3d, v_px, v) for v, v_px in zip(frame_world, frame_px)]
    # for coord in frame_debug:
    #     new_obj = bpy.data.objects.new('empty', None)
    #     new_obj.location = coord
    #     bpy.context.scene.collection.objects.link(new_obj)

    # for i in range(11):
    #     frame_tmp = [obj.matrix_world @ (0.1 * i * v) for v in frame_local]
    #     frame_debug = [region_2d_to_location_3d(region, rv3d, v_px, v) for v, v_px in zip(frame_tmp, frame_px)]
    #     for coord in frame_debug:
    #         new_obj = bpy.data.objects.new('empty', None)
    #         new_obj.location = coord
    #         bpy.context.scene.collection.objects.link(new_obj)

    return frame_px


def select_camera_border(camera, context, area_region_rv3d=None):
    if area_region_rv3d is None:
        area_region_rv3d = view3d_find()
        # print(area_region_rv3d)
    area, region, rv3d = area_region_rv3d

    # update viewport
    override = bpy.context.copy()
    override['area'] = area
    override['region'] = region
    rv3d.view_perspective = 'CAMERA'
    # rv3d.view_camera_offset = (0.0, 0.0)
    # rv3d.view_camera_zoom = -30
    bpy.ops.view3d.view_center_camera(override)
    context.view_layer.update()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    frame_px = view3d_camera_border(bpy.context.scene, area_region_rv3d)

    if None in frame_px:
        return False

    xmin = min([v[0] for v in frame_px])
    xmax = max([v[0] for v in frame_px])
    ymin = min([v[1] for v in frame_px])
    ymax = max([v[1] for v in frame_px])
    # print('border', xmin, xmax, ymin, ymax)

    override = context.copy()
    override['area'] = area
    override['region'] = region
    bpy.ops.view3d.select_box(override, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, mode='ADD')

    return True


if __name__ == '__main__':
    blender_path = bpy.path.abspath("//")
    parent_path = os.path.abspath(os.path.join(blender_path, os.pardir))

    total_time = time()

    input_image_dir = ''
    print(input_image_dir)

    # object_list = None  # get objects from blend file

    # main(input_image_dir, frames_list=frames_list, frames_step=frames_step, object_list=object_list)

    # -------------------------------------------------------------
    # batch processing
    context = bpy.context
    scene = context.scene

    render_samples = 4

    # frames_list = None
    frames_step = 1
    # frames_step = 10
    # frames_step = 10  # range() step for setting frames
    # frames_step = 100

    frames_list = None  # get frames from blend file
    # frames_list = [1]
    # frames_list = [1, 50]
    # frames_list = [21]
    # frames_list = [1, 20, 40, 60, 80]
    # frames_list = [1, 20, 40, 60, 80, 100, 120, 140]

    # remove materials
    object_list = [o for o in scene.objects if o.type == 'MESH']
    for ob in object_list:
        ob.active_material_index = 0
        for i in range(len(ob.material_slots)):
            bpy.ops.object.material_slot_remove({'object': ob})

    # override objects here for texture mapping
    # object_list = [scene.objects['bed'], scene.objects['painting']]
    # object_list = [ scene.objects[name] for name in object_list]

    # batch loop

    n_object = len(object_list)
    batch_size = 1
    # split = 10
    # batch_size = len(object_list) // split
    start = 0
    end = 0
    s_time = time()
    while end < n_object:
        end = min(start + batch_size, n_object)
        _object_list = object_list[start: end]
        main(input_image_dir, frames_list=frames_list, frames_step=frames_step, object_list=_object_list,
             render_samples=render_samples, save_mainfile=True, show_aggregated=False, debug=False)
        start = end

        print(f'[{end}/{n_object}] elp: {time() - s_time} ...')
        s_time = time()

    # -----
    # assign material
    context.view_layer.objects.active = None
    for obj in bpy.data.objects:
        obj.select_set(False)

    # assign material
    object_list = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    for obj in object_list:
        aggregated_image = bpy.data.images.get(f'{obj.name}.aggregated_texture', None)
        if aggregated_image is None:
            # bpy.data.objects.remove(obj)
            continue
        material = create_mat_aggregated(f'{obj.name}.aggregated', f'unwrapped_uvmap.{obj.name}', aggregated_image)
        # material = create_mat_aggregated_bsdf(f'{obj.name}.aggregated_bsdf', f'unwrapped_uvmap.{obj.name}', aggregated_image)

        context.view_layer.objects.active = obj  # set active object
        obj.select_set(True)  # select object
        bpy.ops.object.mode_set(mode='EDIT')  # entering edit mode
        bpy.ops.mesh.select_all(action='SELECT')  # select all objects elements
        assign_material_to_object(obj, material)  # assign material to object

        bpy.ops.mesh.select_all(action='DESELECT')  # deselect all objects elements
        bpy.ops.object.mode_set(mode='OBJECT')  # exiting edit mode
        context.view_layer.objects.active = None
        obj.select_set(False)
    print('assigned material to objects')

    bpy.ops.wm.save_mainfile()
    
    print(f'total elp: {time() - total_time} ...')

# # apply blank material for other object
# mat_blank = create_mat_blank()
# other_object_list = set([o for o in scene.objects if o.type == 'MESH']) - {obj}
# # other_object_list = set([object_dict['object'] for object_dict in object_dict_list])  - {obj}
# for other_obj in other_object_list:
#     if other_obj.data.materials:
#         # assign to 1st material slot
#         old_mat = other_obj.data.materials[0]
#         other_obj.data.materials[0] = mat_blank
#         other_obj.data.materials.append(old_mat)
#     else:
#         other_obj.data.materials.append(mat_blank)
#     # context.view_layer.objects.active = other_obj  # set active object
#     # other_obj.select_set(True)  # select object
#     # bpy.ops.object.mode_set(mode='EDIT')  # entering edit mode
#     # bpy.ops.mesh.select_all(action='SELECT')  # select all objects elements
#     # assign_material_to_object(other_obj, mat_blank)  # assign material to object
#     # bpy.ops.mesh.select_all(action='DESELECT')  # deselect all objects elements
#     # bpy.ops.object.mode_set(mode='OBJECT')  # exiting edit mode
#     # context.view_layer.objects.active = None
#     # other_obj.select_set(False)


# remove blank material for other object
# for other_obj in other_object_list:
#     if len(other_obj.data.materials) > 1:
#         # assign to 1st material slot
#         old_mat = other_obj.data.materials.pop()
#         other_obj.data.materials[0] = old_mat
#     else:
#         other_obj.data.materials.pop()
# context.view_layer.objects.active = other_obj  # set active object
# other_obj.select_set(True)  # select object
# bpy.ops.object.mode_set(mode='EDIT')  # entering edit mode
# bpy.ops.mesh.select_all(action='SELECT')  # select all objects elements
# remove_material_from_object(other_obj, mat_blank)  # remove material from object
# bpy.ops.mesh.select_all(action='DESELECT')  # deselect all objects elements
# bpy.ops.object.mode_set(mode='OBJECT')  # exiting edit mode
# context.view_layer.objects.active = None
# other_obj.select_set(False)
