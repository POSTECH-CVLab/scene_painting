import bpy
import random
import math

import os
import numpy as np

tx = 0.0
ty = 0.0
tz = 0.0

rx = 0.0
ry = 0.0
rz = 0.0

if 'blender_set_008' in bpy.data.filepath:
    theta_min = 0.22
    theta_max = 0.45
    fov = 80
    r = 1
elif 'living_room' in bpy.data.filepath:
    theta_min = 0.3
    theta_max = 0.55
    fov = 80
    r = 1
elif 'kitchen' in bpy.data.filepath:
    theta_min = 0.35
    theta_max = 0.55
    fov = 50   
    r = 1.8
elif 'bedroom' in bpy.data.filepath:
    theta_min = 0.3
    theta_max = 0.55
    fov = 80   
    r = 1.8
elif 'office' in bpy.data.filepath:
    theta_min = 0.3
    theta_max = 0.55
    fov = 80   
    r = 1
else:
    theta_min = 0.3
    theta_max = 0.55
    fov = 80
    r = 1


num_frames = 10000
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = num_frames


scene = bpy.data.scenes["Scene"]

# Set render resolution
scene.render.resolution_x = 256
scene.render.resolution_y = 256

# Set camera rotation in euler angles
scene.camera.rotation_mode = 'XYZ'

target = bpy.data.objects['Empty']
target_x = target.location.x
target_y = target.location.y
target_z = target.location.z

# Set camera fov in degrees
# fov = random.uniform(80, 100)

scene.camera.data.angle = fov*(math.pi/180.0)


xyz_list = []
for i in range(num_frames):
#    scene.camera.rotation_euler[0] = rx*(math.pi/180.0)
#    scene.camera.rotation_euler[1] = ry*(math.pi/180.0)
#    scene.camera.rotation_euler[2] = rz*(math.pi/180.0)

    # TODO: Set camera field of view (random sampling)    

    theta = random.uniform(theta_min, theta_max) * math.pi * 1.0
    phi = random.uniform(0, 1) * math.pi * 2
    
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    
    # Set camera translation
    scene.camera.location.x = x + target_x
    scene.camera.location.y = y + target_y
    scene.camera.location.z = z + target_z
    
    xyz_list.append([x,y,z])
        
    scene.camera.keyframe_insert(data_path="location", frame=i+1)

xyz_list = np.array(xyz_list)
print('x:', np.min(xyz_list[:,0]), np.max(xyz_list[:,0]))
print('y:', np.min(xyz_list[:,1]), np.max(xyz_list[:,1]))
print('z:', np.min(xyz_list[:,2]), np.max(xyz_list[:,2]))

dir_path = bpy.data.filepath[:-6] + '_raw_data/labels'
os.makedirs(dir_path, exist_ok=True)

dir_path = bpy.data.filepath[:-6] + '_raw_data/depths'
os.makedirs(dir_path, exist_ok=True)


## switch on nodes and get reference
#bpy.context.scene.use_nodes = True
#tree = bpy.context.scene.node_tree

## clear default nodes
#for node in tree.nodes:
#    tree.nodes.remove(node)

## create input image node
#image_node = tree.nodes.new(type='CompositorNodeImage')
#image_node.image = bpy.data.images['YOUR_IMAGE_NAME']
#image_node.location = 0,0

## create output node
#comp_node = tree.nodes.new('CompositorNodeComposite')   
#comp_node.location = 400,0

## link nodes
#links = tree.links
#link = links.new(image_node.outputs[0], comp_node.inputs[0])
