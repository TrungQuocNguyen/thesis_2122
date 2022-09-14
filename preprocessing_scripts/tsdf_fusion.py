import argparse
import time

import cv2
import numpy as np

import fusion_pytorch as fusion

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}


def main(args):
  if args.example == 'py':
    print("Using vanilla PyTorch.")
  else:
    pass

  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 880+1
  cam_intr = np.array([[289.8382, 0.0, 159.5616], [0.0, 290.1292, 119.5618], [0.0, 0.0, 1.0]])
  vol_bnds = np.zeros((3,2))
  for i in range(0,n_imgs,10):
    # Read depth image and camera pose
    depth_im = cv2.imread("/mnt/raid/tnguyen/scannet_2d3d/scene0552_00/depth/" + str(i) + ".png",-1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    cam_pose = np.loadtxt("/mnt/raid/tnguyen/scannet_2d3d/scene0552_00/pose/" + str(i) + ".txt")  # 4x4 rigid transformation matrix

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)  #np array of size (3,5) consists of 5 points, which form a pyramid with the top is optical center 
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #

  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, 0.05, fusion.integrate)

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  times = []
  for i in range(0,n_imgs,10):
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread("/mnt/raid/tnguyen/scannet_2d3d/scene0552_00/color/"+ str(i) + ".jpg"), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread("/mnt/raid/tnguyen/scannet_2d3d/scene0552_00/depth/" + str(i) + ".png",-1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    cam_pose = np.loadtxt("/mnt/raid/tnguyen/scannet_2d3d/scene0552_00/pose/" + str(i) + ".txt") 

    # Integrate observation into voxel volume (assume color aligned with depth)
    tic = time.time()
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
    toc = time.time()
    times.append(toc-tic)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  times = [t*TIME_SCALES[args.scale] for t in times]
  print("Average integration time: {:.3f} {}".format(np.mean(times), args.scale))

  # Extract pointcloud
  point_cloud = tsdf_vol.extract_point_cloud()
  fusion.pcwrite("/home/tnguyen/thesis_2122/pc.ply", point_cloud)

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.extract_triangle_mesh()
  fusion.meshwrite("/home/tnguyen/thesis_2122/mesh.ply", verts, faces, norms, colors)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('example', choices=['pycuda', 'py', 'cpp', 'jit', 'cuda'])
  parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='s')
  args = parser.parse_args()
  main(args)