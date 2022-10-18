# From reconstruction to segmentation: Using pretrained features from 3D reconstruction in semantic segmentation
Semantic scene segmentation and reconstruction is two of the most important tasks
in computer vision. Although their tasks are different, they share many things in
common. For example, knowing the occluded objects behind the table is a chair helps
us imagine where and what shape the chair leg is, although it is not visible. As a
result, many works have been dedicated to combine two tasks at once, with the goal
of improving both of them by training them jointly. Such task is called semantic 3D
reconstruction. The joint training has proved effective, but it is difficult to know how
and how much one task influence the other. In this work, we try to separate two tasks
and analyze the relationship between them throuh two main experiments. We first
train a 2D segmentation network, then use it as a feature extractor for training on
reconstruction task. We study the effect segmentation model has by training another 3D
network without using the feature extractor. In the second experiment, we finetune the
reconstruction net to scene segmentation task and check if the influence in the opposite
direction also holds.

<img src="images/3d_recon.png" width="600"/>


<img src="images/3d_semseg.png" width="600"/>

## Dataset preparation
Create a folder `/home/scannet/scans` and download the full scannet dataset to it. The structure of dataset should be as follows: 
```
   scans/
   |--scene0000_00/
      |--scene0000_00.sens
      |--label-filt/
         |--[framenum].png
             ⋮
   |--scene0000_01/
      ⋮
   ```
Run the following line of code to extract color, depth frames, camera pose and label images to folder `scannet_2d3d`:
```
python preprocessing_scripts/prepare_2d_data.py --scannet_path /home/scannet/scans --output_path scannet_2d3d --export_label_images --frame_skip 10 --label_map_file [path to scannetv2-labels.combined.tsv]
```
The structure of folder `scannet_2d3d` should now be: 
```
   |--scene0000_00/
      |--color/
         |--[framenum].jpg
             ⋮
      |--depth/
         |--[framenum].png
             ⋮
      |--pose/
         |--[framenum].txt
             ⋮
      |--label/
         |--[framenum].png
             ⋮
   |--scene0000_01/
      ⋮
   ```
