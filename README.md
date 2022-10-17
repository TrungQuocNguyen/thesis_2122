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
