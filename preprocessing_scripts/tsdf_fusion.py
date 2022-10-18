import argparse
import os
import cv2
import numpy as np

import fusion_pytorch as fusion

def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    scenes = sorted(os.listdir(args.images_path))  # don't count for 'data_chunks' folder which contains data for subvolume
    cam_intr = np.array([[289.8382, 0.0, 159.5616], [0.0, 290.1292, 119.5618], [0.0, 0.0, 1.0]]) # for [320, 240]
    print("Total number of scenes: %d"%(len(scenes)))
    for scene in scenes: 
        scene_path = os.path.join(args.images_path, scene)
        n_imgs = len(os.listdir(os.path.join(scene_path, 'pose')))
        vol_bnds = np.zeros((3,2))
        print("Processing scene %s which contains %d frames"%(scene, n_imgs))
        for i in range(0,n_imgs):
            # Read depth image and camera pose
            depth_im = cv2.imread(os.path.join(scene_path, "depth", str(i*10) + ".png"),-1).astype(float)
            depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
            cam_pose = np.loadtxt(os.path.join(scene_path, "pose", str(i*10) + ".txt"))  # 4x4 rigid transformation matrix
            if np.isinf(cam_pose).any(): 
                continue
            # Compute camera view frustum and extend convex hull
            view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)  #np array of size (3,5) consists of 5 points, which form a pyramid with the top is optical center 
            vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))

        tsdf_vol = fusion.TSDFVolume(vol_bnds, 0.02, fusion.integrate)

        for i in range(0,n_imgs):
            print("Fusing frame %d/%d"%(i*10, n_imgs*10))

            # Read RGB-D image and camera pose
            color_image = cv2.cvtColor(cv2.imread(os.path.join(scene_path, "color", str(i*10) + ".jpg")), cv2.COLOR_BGR2RGB)
            depth_im = cv2.imread(os.path.join(scene_path, "depth", str(i*10) + ".png"),-1).astype(float)
            depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
            cam_pose = np.loadtxt(os.path.join(scene_path, "pose", str(i*10) + ".txt"))  # 4x4 rigid transformation matrix
            if np.isinf(cam_pose).any(): 
                continue
            # Integrate observation into voxel volume (assume color aligned with depth)
            tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
        
        print("Saving to mesh.ply...")
        verts, faces, norms, colors = tsdf_vol.extract_triangle_mesh()
        output_scene_path = os.path.join(args.output_path, scene)
        if not os.path.isdir(output_scene_path):
            os.makedirs(output_scene_path)
        fusion.meshwrite(os.path.join(output_scene_path, scene + ".ply"), verts, faces, norms, colors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', required=True, help='path to rgb images, which will be fused into TSDF volume') # /mnt/raid/tnguyen/scannet_2d3d
    parser.add_argument('--output_path', required=True, help='path to output folder')# /mnt/raid/tnguyen/scannet_2d3d, same as images_path
    args = parser.parse_args()
    main(args)