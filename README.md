# scene_painting

## 1. Make dataset
- Download SceneNet files
- Open a scene layout file in Blender
  - Set blender filename: .../{filename}.blend
- Labeling
  - Assign a class label of each object by filling pass index
  - (Optional) Use 'pre_process/script_labeling_scene.py'
- Render setting
  - Render properties
    - Render engine - cycles / GPU
  - Output properties
    - Resolution: 256x256
   - View Layer properties
    - passes : z, object index
- Add Plane to compensate empty space
- Make camera viewpoints (animation)
  - Set camera viewpoints of keyframes
  - (optional) Use 'pre_process/script_random_camera.py'
- Save camera parameters (focal length, principal point, extrinsic)
  - Use 'pre_process/script_save_camera_params.py'
  - Output: .../{filename}_raw_data/cameras
- Compositing
  - [Reference](http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender)
  - Depth
    - Set save path: .../{filename}_raw_data/depths
    - File Format: OpenEXR
  - IndexOB (label)
    - Set save path: .../{filename}_raw_data/labels
    - Divide node values by 255
    - File Format: PNG, Color: BW, Color Depth: 8, compression 15 
- Rendering
  - Ctrl + F12
  - Output: .../{filename}_raw_data/depths
  - Output: .../{filename}_raw_data/labels
- Make train dataset
  - Enter pre_process directory
  - python make_dataset.py --data_root .../{filename}_raw_data
  - Output: .../{filename}/coordinate_images
  - Output: .../{filename}/labels_ade20k
  - Output: .../{filename}/labels_scenenet
  - Output: .../{filename}/ade_to_scenenet.pickle
  - Output: .../{filename}/stats.npz
- (Optional) Make test dataset
  - Assume that '.../{testset}_raw_data' exists.
  - Make test dataset using data properties of train dataset (ade_to_scenenet.pickle and stats.npz)
  - python make_dataset.py --data_root .../{testset}_raw_data --is_testset --stats_path .../{filename}/stats.npz --ade_to_scenenet_path .../{filename}/ade_to_scenenet.pickle


## 2. Training
[Training](https://github.com/jbjeong/scene_painting/tree/master/training)

## 3. Texture mapping
- Open the scene layout file used to make the dataset in Blender
  - Use blender script: 'pre_process/script_texture_mapping.py'
  - Change the generated image path and execute the script

