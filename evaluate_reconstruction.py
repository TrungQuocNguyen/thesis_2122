import os
import math
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import pyrender
import trimesh
from torchvision import transforms as T
import open3d as o3d


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1
    
    Args:
        nx3 np.array's

    Returns:
        ([indices], [distances])
    
    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances

class Renderer():
    """OpenGL mesh renderer 
    
    Used to render depthmaps from a mesh for 2d evaluation
    """
    def __init__(self, height=256, width=328):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)#, self.render_flags) 

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R =  np.array([[1, 0, 0],
                       [0, c, -s],
                       [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose@axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()
def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def resize_crop_image(image, new_image_dims):
        #image: [240, 320]
        #new_image_dims: [328, 256]
        image_dims = [image.size[0], image.size[1]] # [320, 240]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = T.Resize([new_image_dims[1], resize_width], interpolation=T.InterpolationMode.NEAREST)(image)
        image = T.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        return image

def load_pose(filename):
        pose = np.zeros((4, 4))
        lines = open(filename).read().splitlines()
        assert len(lines) == 4
        lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
        return np.asarray(lines).astype(np.float32)

def load_depth(file, image_dims):
        #image_dims: [328, 256]
        depth_image = Image.open(file) # [240, 320]
        # preprocess
        depth_image = resize_crop_image(depth_image, image_dims) # (256, 328)
        depth_image = np.array(depth_image)
        depth_image = depth_image.astype(np.float32) / 1000.0
        return depth_image

def eval_depth(depth_pred, depth_trgt):
    """ Computes 2d metrics between two depth maps
    
    Args:
        depth_pred: mxn np.array containing prediction
        depth_trgt: mxn np.array containing ground truth

    Returns:
        Dict of metrics
    """
    mask1 = depth_pred>0 # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt<10) * (depth_trgt>0) * mask1

    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred-depth_trgt)
    abs_rel = abs_diff/depth_trgt
    sq_diff = abs_diff**2
    sq_rel = sq_diff/depth_trgt
    sq_log_diff = (np.log(depth_pred)-np.log(depth_trgt))**2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype('float')
    r2 = (thresh < 1.25**2).astype('float')
    r3 = (thresh < 1.25**3).astype('float')

    metrics = {}
    metrics['AbsRel'] = np.mean(abs_rel)
    metrics['AbsDiff'] = np.mean(abs_diff)
    metrics['SqRel'] = np.mean(sq_rel)
    metrics['RMSE'] = np.sqrt(np.mean(sq_diff))
    metrics['LogRMSE'] = np.sqrt(np.mean(sq_log_diff))
    metrics['r1'] = np.mean(r1)
    metrics['r2'] = np.mean(r2)
    metrics['r3'] = np.mean(r3)
    metrics['complete'] = np.mean(mask1.astype('float'))

    return metrics

def main(args):
    threshold_3d = 0.05
    key_names_3d = ['Accuracy', 'Completeness', 'Precision', 'Recall', 'F1_score'] 
    key_names_2d = ['AbsRel', 'AbsDiff', 'SqRel', 'RMSE', 'LogRMSE', 'r1', 'r2', 'r3', 'complete']
    metrics = {m:[] for m in (key_names_3d + key_names_2d)}

    depth_shape = [328, 256]
    intrinsics = make_intrinsic(308.7996, 309.4712, 163.5631, 127.5659)  #intrinsic camera for image size [328, 256]
    base_dir = Path(args.base_dir)
    file_path = base_dir / 'scannetv2_val.txt'
    with open(file_path) as f:
        lines = f.readlines()
    cleanlines = [line.strip() for line in lines]
    for scene_name in cleanlines: #loop over each scene
        print('Processing %s...'%(scene_name))
        renderer = Renderer()
        pred_path = base_dir / args.model_type / (scene_name + '.ply')
        ############################### Benchmarking using 3D metrics #####################################
        if 'scannet' in args.gt_type: # evalute with GT from scannet: 
            gt_path = base_dir / args.gt_type / (scene_name + '_vh_clean_2.ply')
        else: # evaluate with GT from self-generated TSDF 
            gt_path = base_dir / args.gt_type / (scene_name + '.ply')

        v_gt = o3d.io.read_point_cloud(str(gt_path))
        v_sampled_gt = v_gt.voxel_down_sample(0.02)
        v_sampled_gt = np.asarray(v_sampled_gt.points)

        v_pred = o3d.io.read_point_cloud(str(pred_path))
        v_sampled_pred = v_pred.voxel_down_sample(0.02)
        v_sampled_pred = np.asarray(v_sampled_pred.points)

        _, dist1 = nn_correspondance(v_sampled_gt, v_sampled_pred)# accuracy, list of raw python number
        _, dist2 = nn_correspondance(v_sampled_pred, v_sampled_gt)# completeness, list of raw python number

        dist1 = np.array(dist1)
        dist2 = np.array(dist2)

        acc = np.mean(dist1)
        compl = np.mean(dist2)
        prec = np.mean((dist1<threshold_3d).astype('float'))
        rec = np.mean((dist2<threshold_3d).astype('float'))
        f1 = 2 * prec * rec / (prec + rec)

        metrics['Accuracy'].append(acc)
        metrics['Completeness'].append(compl)
        metrics['Precision'].append(prec)
        metrics['Recall'].append(rec)
        metrics['F1_score'].append(f1)

        ############################### Benchmarking using 2D metrics #####################################

        mesh = trimesh.load(pred_path, process=False)
        mesh_opengl = renderer.mesh_opengl(mesh)
        scan_dir = base_dir / 'RGB_images' / scene_name
        depth_dir = scan_dir / 'depth'
        pose_dir  = scan_dir / 'pose'

        pose_files = sorted(os.listdir(pose_dir), key=lambda f: int(os.path.splitext(f)[0]))
        invalid_pose = 0
        for i, f in enumerate(pose_files): #loop over each frame in a scene
            pose_path = pose_dir / f
            depth_path = depth_dir / f'{Path(f).stem}.png'
            pose = load_pose(pose_path)
            if np.isinf(pose).any(): 
                #print('Omitting frame %d because of invalid pose'%(i))
                invalid_pose +=1
                continue
            depth_trgt = load_depth(depth_path, depth_shape)
            _, depth_pred = renderer(depth_shape[1], depth_shape[0], intrinsics, pose, mesh_opengl)
            temp = eval_depth(depth_pred, depth_trgt)
            if i==0:
                metrics_depth = temp
            else:
                metrics_depth = {key:value+temp[key] for key, value in metrics_depth.items()}

        metrics_depth = {key:value/(len(pose_files)-invalid_pose) 
                     for key, value in metrics_depth.items()}
        for k in key_names_2d: 
            metrics[k].append(metrics_depth[k])

    for k in (key_names_3d + key_names_2d): 
        v = np.nanmean(np.array(metrics[k]))
    
        print('%10s %0.3f'%(k, v))

if __name__ =='__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', help='Path to folder containing GT and predicted mesh (contain everything)') # /home/trung/test_inference_whole_scenes/from_ScanNet_GT
    parser.add_argument('--gt_type', help='type of ground truth to evaluate, choose between: gt_from_scannet (Evaluate from GT of ScanNet) or gt_from_tsdf (Evaluate from GT of TSDF fusion)') #  gt_from_scannet
    parser.add_argument('--model_type', help='type of model to evaluate') #  predicted_scenes_RGB_pos8

    args = parser.parse_args()
    print(args)
    main(args)