import h5py
import glob
from pathlib import Path
import numpy as np 
def main(): 
    n_samples = 120100
    subvol_size = tuple([32, 32, 64])
    num_nearest_images = 5
    start_ndx = 0
    split = 'train'
    with h5py.File('/mnt/raid/tnguyen/scannet_2d3d/data_chunks/' + split +'.hdf5', 'w') as outfile: 
        outfile.create_dataset('x', (n_samples,) + subvol_size, dtype=np.float32)
        # label subvolume
        outfile.create_dataset('y', (n_samples,) + subvol_size, dtype=np.int16)
        # id of the scene that the volume came from (0000, 0002 ..)
        outfile.create_dataset('scene_id', (n_samples,), dtype=np.uint16)
        # id of the scan within the scene: 00, 01, 02..
        outfile.create_dataset('scan_id', (n_samples,), dtype=np.uint8)
        # world to grid transformation for this subvolume
        outfile.create_dataset('world_to_grid', (n_samples, 4, 4), dtype=np.float32)
        # indices of the corresponding frames
        outfile.create_dataset('frames', (n_samples, num_nearest_images), dtype=np.int16)
        
        for name in sorted(glob.glob('/mnt/raid/tnguyen/scannet_2d3d/data_chunks/' + split + '_chunks/' + split + '*'), key = lambda a: int(Path(a).stem[12:])): 
            print(' reading file %s' %(Path(name).stem))
            file = h5py.File(name, 'r')
            num_chunks = len(file['scene_id'])

            outfile['frames'][start_ndx:start_ndx + num_chunks] = file['frames'][:]
            outfile['x'][start_ndx:start_ndx + num_chunks] = file['x'][:]
            outfile['y'][start_ndx:start_ndx + num_chunks] = file['y'][:]
            outfile['scene_id'][start_ndx:start_ndx + num_chunks] = file['scene_id'][:]
            outfile['scan_id'][start_ndx:start_ndx + num_chunks] = file['scan_id'][:]
            outfile['world_to_grid'][start_ndx:start_ndx + num_chunks] = file['world_to_grid'][:]
            start_ndx = start_ndx + num_chunks

if __name__ == '__main__':
    main()