import argparse
from prep_backproj_data import create_datasets
from tqdm import tqdm
import h5py
import random 

def main(args): 
    assert args.in_path != args.out_path
    out_samples = 4
    with h5py.File(args.in_path, 'r') as f:
        subvol_size = f['x'][0].shape
        in_samples = len(f['x']) 
        num_nearest_images = len(f['frames'][0])

        with h5py.File(args.out_path, 'w') as outf:
            create_datasets(outf, out_samples, subvol_size, num_nearest_images)

            for index in range(out_samples):

                ndx = random.randint(0,in_samples-1)
                x, y, world_to_grid = f['x'][ndx], f['y'][ndx], f['world_to_grid'][ndx] 
                # these wont change
                scan_id, scene_id, frames = f['scan_id'][ndx], f['scene_id'][ndx], f['frames'][ndx]

                outf['scan_id'][index] = scan_id
                outf['scene_id'][index] = scene_id
                outf['frames'][index] = frames
                outf['x'][index] = x
                outf['y'][index] = y
                outf['world_to_grid'][index] = world_to_grid
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('in_path', help='Path to input h5 file')
    p.add_argument('out_path', help='Path to output h5 file')

    args = p.parse_args()

    main(args)