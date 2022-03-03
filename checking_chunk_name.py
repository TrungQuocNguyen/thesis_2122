import os
scene_names = set()
data_location = "/home/tnguyen/thesis_2122/experiments/filelists/ScanNet/v1/test.txt"
datalist = open(data_location, 'r')
scenes = [x.strip() for x in datalist.readlines()]
#with open("experiments/filelists/ScanNet/v2/chunk_from_one_val_scene/chunk.txt", "w") as f:
for scene in scenes:
    scene_name = os.path.basename(scene).split('__')[0]
    scene_names.add(scene_name)
        #if scene_name == 'scene0377_00': 
        #    f.write(scene + '\n')
#f.close()
scene_names = list(scene_names)
scene_names.sort()
print(scene_names)
print(len(scene_names))