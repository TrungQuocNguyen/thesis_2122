import os
import numpy as np 
import torch 
import point_cloud_utils as pcu
from scipy.spatial.distance import cdist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
threshold = 0.05


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1
    
    Args:
        nx3 np.array's

    Returns:
        [distances]
    
    """
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return distances
    for vert in verts2: 
        dist = np.min(cdist(np.expand_dims(vert, 0), verts1))
        distances.append(dist)
    return distances

def nn_correspondance_gpu(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1
    
    Args:
        nx3 np.array's

    Returns:
        [distances]
    
    """
    verts1 = torch.from_numpy(verts1).to(device)
    verts2 = torch.from_numpy(verts2).to(device)
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return distances
    distances, _ = torch.min(torch.cdist(verts2, verts1), dim = 1)
    return distances.tolist()

def main(): 
    file_path = '/mnt/raid/tnguyen/scannetv2_val.txt'
    gt_path = '/mnt/raid/tnguyen/scannet_2d3d'
    pred_path = '/home/tnguyen/thesis_2122/jupyter_notebook/from_TSDF/predicted_scenes_RGB'
    with open(file_path) as f:
        lines = f.readlines()
    cleanlines = [line.strip() for line in lines]

    accuracy = []
    completeness = []
    precision = []
    recal = []
    fscore = []
    for scene_name in cleanlines: 
        print('Processing %s...'%(scene_name))
        gt_file = os.path.join(gt_path, scene_name, scene_name + '.ply')
        pred_file = os.path.join(pred_path, scene_name + '.ply')
    
        v_gt, _ = pcu.load_mesh_vf(gt_file)
        v_sampled_gt, _, _ = pcu.downsample_point_cloud_voxel_grid(0.05, v_gt)

        v_pred, _ = pcu.load_mesh_vf(pred_file)
        v_sampled_pred, _, _ = pcu.downsample_point_cloud_voxel_grid(0.05, v_pred)

        ######################## Performing correspondence on GPU or CPU ########################################
        print('Calculate correspondence between GT and pred...')
        v_sampled_gt = v_sampled_gt.astype(np.float64)
        v_sampled_pred = v_sampled_pred.astype(np.float64)
    
        dist1 = nn_correspondance(v_sampled_gt, v_sampled_pred)# accuracy, list of raw python number
        dist2 = nn_correspondance(v_sampled_pred, v_sampled_gt)# completeness, list of raw python number
        print('Done.')
        ##########################################################################################################

        dist1 = np.array(dist1)
        dist2 = np.array(dist2)

        acc = np.mean(dist1)
        compl = np.mean(dist2)
        prec = np.mean((dist1<threshold).astype('float'))
        rec = np.mean((dist2<threshold).astype('float'))
        f1 = 2 * prec * rec / (prec + rec)

        accuracy.append(acc)
        completeness.append(compl)
        precision.append(prec)
        recal.append(rec)
        fscore.append(f1)
    print('Mean of Accuracy is: %.4f'%(np.mean(accuracy)))
    print('Mean of Completeness is: %.4f'%(np.mean(completeness)))
    print('Mean of Precision is: %.4f'%(np.mean(precision)))
    print('Mean of Recal is: %.4f'%(np.mean(recal)))
    print('Mean of F1 score is: %.4f'%(np.mean(fscore)))
    print()
    print('Median of Accuracy is: %.4f'%(np.median(accuracy)))
    print('Median of Completeness is: %.4f'%(np.median(completeness)))
    print('Median of Precision is: %.4f'%(np.median(precision)))
    print('Median of Recal is: %.4f'%(np.median(recal)))
    print('Median of F1 score is: %.4f'%(np.median(fscore)))
if __name__ =='__main__': 
    main()


