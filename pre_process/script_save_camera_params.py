import bpy
from mathutils import Matrix

import os
from mathutils import *

import numpy as np


# https://blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model/38189#38189
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  ,  alpha_v, v_0),
        (    0  ,    0,      1 )))
    return K

#prefix_image = "/home/jbjeong/projects/scene3d/test_images/"

#def get_camera_pose(cameraName, objectName, scene, frameNumber):
#    if not os.path.exists(prefix_pose):
#        os.makedirs(prefix_pose)
#
#    # OpenGL to Computer vision camera frame convention
#    M = Matrix().to_4x4()
#    M[1][1] = -1
#    M[2][2] = -1
#
#    cam = bpy.data.objects[cameraName]
#    object_pose = bpy.data.objects[objectName].matrix_world
#
#    #Normalize orientation with respect to the scale
#    object_pose_normalized = object_pose.copy()
#    object_orientation_normalized = object_pose_normalized.to_3x3().normalized()
#    for i in range(3):
#        for j in range(3):
#            object_pose_normalized[i][j] = object_orientation_normalized[i][j]
#
#    camera_pose = M*cam.matrix_world.inverted()*object_pose_normalized
#
#    print("camera_pose:\n", np.array(camera_pose))
#    print("rotation:\n", np.array(camera_pose.to_3x3()))
#    print("translation:\n", np.array(camera_pose.to_translation()))
#  
#    filename = prefix_pose + cameraName + "_%03d" % frameNumber + ".npz"
#    
#    np.savez(filename, 
#             matrix=np.array(camera_pose),
#             rotation=np.array(camera_pose.to_3x3()), 
#             translation=np.array(camera_pose.to_translation()))
#  
#    filename = prefix_pose + cameraName + "_%03d" % frameNumber + ".txt"
#    with open(filename, 'w') as f:
#        f.write(str(camera_pose[0][0]) + " ")
#        f.write(str(camera_pose[0][1]) + " ")
#        f.write(str(camera_pose[0][2]) + " ")
#        f.write(str(camera_pose[0][3]) + " ")
#        f.write("\n")
#
#        f.write(str(camera_pose[1][0]) + " ")
#        f.write(str(camera_pose[1][1]) + " ")
#        f.write(str(camera_pose[1][2]) + " ")
#        f.write(str(camera_pose[1][3]) + " ")
#        f.write("\n")
#
#        f.write(str(camera_pose[2][0]) + " ")
#        f.write(str(camera_pose[2][1]) + " ")
#        f.write(str(camera_pose[2][2]) + " ")
#        f.write(str(camera_pose[2][3]) + " ")
#        f.write("\n")
#
#        f.write(str(camera_pose[3][0]) + " ")
#        f.write(str(camera_pose[3][1]) + " ")
#        f.write(str(camera_pose[3][2]) + " ")
#        f.write(str(camera_pose[3][3]) + " ")
#        f.write("\n")
#    return

#def get_camera_pose(cameraName, scene, frameNumber):
#    if not os.path.exists(prefix_pose):
#        os.makedirs(prefix_pose)
#
#    # OpenGL to Computer vision camera frame convention
##    M = Matrix().to_4x4()
##    M[1][1] = -1
##    M[2][2] = -1
#
#    cam = bpy.data.objects[cameraName]
#    camera_pose = cam.matrix_world
#    #camera_pose = M*cam.matrix_world.inverted()*object_pose_normalized
#    
#    print("camera_pose:\n", np.array(camera_pose))
#    print("rotation:\n", np.array(camera_pose.to_3x3()))
#    print("translation:\n", np.array(camera_pose.to_translation()))
#  
#    filename = prefix_pose + cameraName + "_%03d" % frameNumber + ".npz"
#    
#    np.savez(filename, 
#             matrix=np.array(camera_pose),
#             rotation=np.array(camera_pose.to_3x3()), 
#             translation=np.array(camera_pose.to_translation()))


# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cameraName, scene, frameNumber, prefix_pose):
    
    cam = bpy.data.objects[cameraName]
    
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
            ((1, 0,  0),
             (0, -1, 0),
             (0, 0, -1)))
    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam
    
    print("rotation:\n", np.array(R_world2cv))
    print("translation:\n", np.array(T_world2cv))  
    filename = os.path.join(prefix_pose, cameraName + "_%04d" % frameNumber + ".npz")
    
    np.savez(filename, 
             rotation=np.array(R_world2cv), 
             translation=np.array(T_world2cv))
    
    # put into 3x4 matrix
    RT = Matrix((
                R_world2cv[0][:] + (T_world2cv[0],),
                R_world2cv[1][:] + (T_world2cv[1],),
                R_world2cv[2][:] + (T_world2cv[2],)
                ))
    return RT


def my_handler(scene, prefix_pose):
    frameNumber = scene.frame_current
    print("\n\nFrame Change", scene.frame_current)
    #get_camera_pose("Camera", "Empty", scene, frameNumber)
    #get_camera_pose("Camera", scene, frameNumber)
    
    get_3x4_RT_matrix_from_blender("Camera", scene, frameNumber, prefix_pose)

if __name__ == "__main__":
    # Insert your camera name below

    prefix_pose = bpy.data.filepath[:-6] + '_raw_data/cameras/'
    os.makedirs(prefix_pose, exist_ok=True)

    scene = bpy.context.scene
    
#    print('hello')   
        
#    step = 1
#    scene.frame_set(step)
#    K = get_calibration_matrix_K_from_blender(bpy.data.objects['Camera'].data)
#    print(K)
#
#    step = 15
#    scene.frame_set(step)
#    K = get_calibration_matrix_K_from_blender(bpy.data.objects['Camera'].data)
#    print(K)

    for step in range(scene.frame_start, scene.frame_end + 1):
        
        print(step)
        # Set render frame
        scene.frame_set(step)
#        print(bpy.data.objects['Camera'].data.angle)
        
        K = get_calibration_matrix_K_from_blender(bpy.data.objects['Camera'].data)
        save_path = os.path.join(prefix_pose, 'intrinsic_%04d.npz' % step)
        np.savez(save_path, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2])
#        print(K)
  
        my_handler(scene, prefix_pose)
        
        #        # Set filename and render
#        if not os.path.exists(prefix_image):
#          os.makedirs(prefix_image)
#
#        scene.render.filepath = (prefix_image + '%04d.png') % step
#        bpy.ops.render.render( write_still=True )
        
    print('bye')
