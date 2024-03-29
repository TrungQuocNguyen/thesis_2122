import numpy as np

from skimage import measure

import torch
CONST_VAL = 256*256

class TSDFVolume:
  """Volumetric TSDF Fusion of RGB-D Images.
  """
  def __init__(self, vol_bnds, voxel_size, integrate_func):
    """Constructor.
    Args:
      vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
        xyz bounds (min/max) in meters.
      voxel_size (float): The volume discretization in meters.
    """
    vol_bnds = np.asarray(vol_bnds)
    assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      print("[!] No GPU detected. Defaulting to CPU.")
      self.device = torch.device("cpu")

    # Define voxel volume parameters
    self._vol_bnds = torch.from_numpy(vol_bnds).float().to(self.device)
    self._voxel_size = float(voxel_size)
    self._sdf_trunc = 5 * self._voxel_size
    self._const = 256*256
    self._integrate_func = integrate_func

    # Adjust volume bounds
    self._vol_dim = torch.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).long()
    self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + (self._vol_dim * self._voxel_size)
    self._vol_origin = self._vol_bnds[:, 0]
    self._num_voxels = torch.prod(self._vol_dim).item()

    # Get voxel grid coordinates
    xv, yv, zv = torch.meshgrid(
      torch.arange(0, self._vol_dim[0]),
      torch.arange(0, self._vol_dim[1]),
      torch.arange(0, self._vol_dim[2]),
    )
    self._vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long().to(self.device) # size (vol_dim[0]*vol_dim[1]*vol_dim[2], 3)

    # Convert voxel coordinates to world coordinates
    self._world_c = self._vol_origin + (self._voxel_size * self._vox_coords)
    self._world_c = torch.cat([
      self._world_c, torch.ones(len(self._world_c), 1, device=self.device)], dim=1) # size (vol_dim[0]*vol_dim[1]*vol_dim[2], 4)

    self.reset()

    print("[*] voxel volume: {} x {} x {}".format(*self._vol_dim))
    print("[*] num voxels: {:,}".format(self._num_voxels))

  def reset(self):
    self._tsdf_vol = torch.ones(*self._vol_dim).to(self.device) # size (vol_dim[0], vol_dim[1], vol_dim[2])
    self._weight_vol = torch.zeros(*self._vol_dim).to(self.device) # size (vol_dim[0], vol_dim[1], vol_dim[2])
    self._color_vol = torch.zeros(*self._vol_dim).to(self.device) # size (vol_dim[0], vol_dim[1], vol_dim[2])

  def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight):
    """Integrate an RGB-D frame into the TSDF volume.
    Args:
      color_im (ndarray): An RGB image of shape (H, W, 3).
      depth_im (ndarray): A depth image of shape (H, W).
      cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
      cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
      obs_weight (float): The weight to assign to the current observation.
    """
    cam_pose = torch.from_numpy(cam_pose).float().to(self.device)
    cam_intr = torch.from_numpy(cam_intr).float().to(self.device)
    color_im = torch.from_numpy(color_im).float().to(self.device)
    depth_im = torch.from_numpy(depth_im).float().to(self.device)
    im_h, im_w = depth_im.shape
    weight_vol, tsdf_vol, color_vol = self._integrate_func(
      color_im,
      depth_im,
      cam_intr,
      cam_pose,
      obs_weight,
      self._world_c,
      self._vox_coords,
      self._weight_vol,
      self._tsdf_vol,
      self._color_vol,
      self._sdf_trunc,
      im_h, im_w,
    )
    self._weight_vol = weight_vol
    self._tsdf_vol = tsdf_vol
    self._color_vol = color_vol

  def extract_point_cloud(self):
    """Extract a point cloud from the voxel volume.
    """
    tsdf_vol = self._tsdf_vol.cpu().numpy()
    color_vol = self._color_vol.cpu().numpy()
    vol_origin = self._vol_origin.cpu().numpy()

    # Marching cubes
    verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size + vol_origin

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / self._const)
    colors_g = np.floor((rgb_vals - colors_b*self._const) / 256)
    colors_r = rgb_vals - colors_b*self._const - colors_g*256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    pc = np.hstack([verts, colors])
    return pc

  def extract_triangle_mesh(self):
    """Extract a triangle mesh from the voxel volume using marching cubes.
    """
    tsdf_vol = self._tsdf_vol.cpu().numpy()
    color_vol = self._color_vol.cpu().numpy()
    vol_origin = self._vol_origin.cpu().numpy()

    # Marching cubes
    verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size + vol_origin

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / self._const)
    colors_g = np.floor((rgb_vals - colors_b*self._const) / 256)
    colors_r = rgb_vals - colors_b*self._const - colors_g*256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    return verts, faces, norms, colors

  @property
  def sdf_trunc(self):
    return self._sdf_trunc

  @property
  def voxel_size(self):
    return self._voxel_size


def rigid_transform(xyz, transform):
  """Applies a rigid transform to an (N, 3) pointcloud.
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = (transform @ xyz_h.T).T
  return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
  """Get corners of 3D camera view frustum of depth image
  """
  im_h = depth_im.shape[0]
  im_w = depth_im.shape[1]
  max_depth = np.max(depth_im)
  max_depth = np.minimum(max_depth, 4.0)
  view_frust_pts = np.array([
    (np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
    (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
    np.array([0,max_depth,max_depth,max_depth,max_depth])
  ])
  view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
  return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
  """Save a 3D mesh to a polygon .ply file.
  """
  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(verts.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property float nx\n")
  ply_file.write("property float ny\n")
  ply_file.write("property float nz\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("element face %d\n"%(faces.shape[0]))
  ply_file.write("property list uchar int vertex_index\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(verts.shape[0]):
    ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
      verts[i,0], verts[i,1], verts[i,2],
      norms[i,0], norms[i,1], norms[i,2],
      colors[i,0], colors[i,1], colors[i,2],
    ))

  # Write face list
  for i in range(faces.shape[0]):
    ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

  ply_file.close()


def pcwrite(filename, xyzrgb):
  """Save a point cloud to a polygon .ply file.
  """
  xyz = xyzrgb[:, :3]
  rgb = xyzrgb[:, 3:].astype(np.uint8)

  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(xyz.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(xyz.shape[0]):
    ply_file.write("%f %f %f %d %d %d\n"%(
      xyz[i, 0], xyz[i, 1], xyz[i, 2],
      rgb[i, 0], rgb[i, 1], rgb[i, 2],
    ))
def integrate(
  color_im,
  depth_im,
  cam_intr,
  cam_pose,
  obs_weight,
  world_c,
  vox_coords,
  weight_vol,
  tsdf_vol,
  color_vol,
  sdf_trunc,
  im_h,
  im_w,
):
  """Integrate an RGB-D frame into the TSDF volume.
  Args: 
    color_im (torch Tensor): An RGB image of shape (H, W, 3).
    depth_im (torch Tensor): A depth image of shape (H, W).
    cam_intr (torch Tensor): The camera intrinsics matrix of shape (3, 3).
    cam_pose (torch Tensor): The camera pose (i.e. extrinsics) of shape (4, 4).
    obs_weight (float): The weight to assign to the current observation.
    world_c (torch Tensor): coordinates of each voxel in world frame, shape (vol_dim[0]*vol_dim[1]*vol_dim[2], 4) (homogeneous coordinates)
    vox_coords (torch Tensor): coordinates of each voxel in grid frame, shape (vol_dim[0]*vol_dim[1]*vol_dim[2], 3)
    weight_vol (torch Tensor): shape (vol_dim[0], vol_dim[1], vol_dim[2])
    tsdf_vol (torch Tensor): shape (vol_dim[0], vol_dim[1], vol_dim[2])
    color_vol (torch Tensor): shape (vol_dim[0], vol_dim[1], vol_dim[2])
    sdf_trunc (float): 5 * _voxel_size, here sdf_trunc = 5*0.05 = 0.25
  """
  # Fold RGB color image into a single channel image
  color_im = torch.floor(color_im[..., 2]*256*256 + color_im[..., 1]*256 + color_im[..., 0]) # torch Tensor of shape [H,W] with values in 0...256x256x256

  # Convert world coordinates to camera coordinates
  world2cam = torch.inverse(cam_pose)
  cam_c = torch.matmul(world2cam, world_c.transpose(1, 0)).transpose(1, 0).float()  # shape (vol_dim[0]*vol_dim[1]*vol_dim[2], 4) (homogeneous coordinates)

  # Convert camera coordinates to pixel coordinates
  fx, fy = cam_intr[0, 0], cam_intr[1, 1]
  cx, cy = cam_intr[0, 2], cam_intr[1, 2]
  pix_z = cam_c[:, 2] # shape (vol_dim[0]*vol_dim[1]*vol_dim[2])
  pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long() # shape (vol_dim[0]*vol_dim[1]*vol_dim[2])
  pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long() # shape (vol_dim[0]*vol_dim[1]*vol_dim[2])

  # Eliminate pixels outside view frustum
  valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0) # shape (vol_dim[0]*vol_dim[1]*vol_dim[2])
  valid_vox_x = vox_coords[valid_pix, 0] # shape (n) n is number of valid voxel, values in 0...vol_dim[0]-1
  valid_vox_y = vox_coords[valid_pix, 1] # shape (n) n is number of valid voxel, values in 0...vol_dim[1]-1
  valid_vox_z = vox_coords[valid_pix, 2] # shape (n), values in 0...vol_dim[2]-1
  valid_pix_y = pix_y[valid_pix] # shape (n), values in 0...im_h-1, can have same values as many voxels correspond to one pixel
  valid_pix_x = pix_x[valid_pix] # shape (n), values in 0...im_w-1, can have same values as many voxels correspond to one pixel
  depth_val = depth_im[pix_y[valid_pix], pix_x[valid_pix]] # shape (n), can have same values as many voxels correspond to one pixel

  # Integrate tsdf
  depth_diff = depth_val - pix_z[valid_pix]
  dist = torch.clamp(depth_diff / sdf_trunc, max=1)
  valid_pts = (depth_val > 0) & (depth_diff >= -sdf_trunc)
  valid_vox_x = valid_vox_x[valid_pts] # shape (k), k is number of voxel that has depth_val >0 and depth_diff >= -sdf_trunc
  valid_vox_y = valid_vox_y[valid_pts] # shape (k)
  valid_vox_z = valid_vox_z[valid_pts] # shape (k)
  valid_pix_y = valid_pix_y[valid_pts] # shape (k)
  valid_pix_x = valid_pix_x[valid_pts] # shape (k)
  valid_dist = dist[valid_pts]  #only take voxels (and corresponding pixels) that has depth diff >= -sdf_trunc. After that, clamp the depth_diff of all voxels that have value depth_diff/sdf_trunc >1 to 1. 
  w_old = weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] # initially 0
  tsdf_vals = tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] # initially 1
  w_new = w_old + obs_weight # 1
  tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (w_old * tsdf_vals + obs_weight*valid_dist) / w_new
  weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new

  # Integrate color
  old_color = color_vol[valid_vox_x, valid_vox_y, valid_vox_z] # initially 0
  old_b = torch.floor(old_color / CONST_VAL) # initially 0
  old_g = torch.floor((old_color-old_b*CONST_VAL) / 256) # initially 0
  old_r = old_color - old_b*CONST_VAL - old_g*256 # initially 0
  new_color = color_im[valid_pix_y, valid_pix_x]
  new_b = torch.floor(new_color / CONST_VAL)
  new_g = torch.floor((new_color - new_b*CONST_VAL) / 256)
  new_r = new_color - new_b*CONST_VAL - new_g*256
  new_b = torch.clamp(torch.round((w_old*old_b + obs_weight*new_b) / w_new), max=255)
  new_g = torch.clamp(torch.round((w_old*old_g + obs_weight*new_g) / w_new), max=255)
  new_r = torch.clamp(torch.round((w_old*old_r + obs_weight*new_r) / w_new), max=255)
  color_vol[valid_vox_x, valid_vox_y, valid_vox_z] = new_b*CONST_VAL + new_g*256 + new_r

  return weight_vol, tsdf_vol, color_vol