#### Nvidia environments:
Nvidia Driver Version: 460.91.03  
CUDA Version: 10.2  
GPU: TITAN RTX 24GB  

#### 0. Prepare environments. 

Install conda environment from environment.yml.  
```$ conda create --name 1915 python==3.8.8 ```  
```$ conda activate 1915 ```  
```$ conda env update --file environment.yml --prune ```  

#### 1. Download the data from this link (https://drive.google.com/file/d/1f7HXSZfh3jdzLuIHQXOc3AtlXknST5qi/view?usp=sharing)

#### 2. Decompress the supple_data_1915.tar
- supple_data
    - ade20k_indoor_size256
    - blender_set_008_test_image
    - checkpoints

#### 3. Place the directories in supple_data to the code_1915.

- code_1915
    - ade20k_indoor_size256
    - blender_set_008_test_image
    - checkpoints
    - ...

#### 4. Train
``` $ bash ./scripts/train_surface_feat_059.sh ```

#### 5. Inference
``` $ bash ./scripts/gen_surface_feat_059_test_iter40000.sh ```
