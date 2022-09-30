import argparse
from prep_backproj_data import create_datasets
from tqdm import tqdm
import h5py
import numpy as np

def main(args): 
    assert args.in_path != args.out_path
    out_samples = 0
    with h5py.File(args.in_path, 'r') as f:
        subvol_size = f['x'][0].shape
        in_samples = len(f['x']) 
        for i in range(in_samples): 
            if (-1 not in f['frames'][i]) and ((f['x'][i]==1).sum()/(32*32*64) > 0.01): 
                out_samples +=1
        print(f'In samples: {in_samples}, out samples: {out_samples}')

        num_nearest_images = len(f['frames'][0])

        with h5py.File(args.out_path, 'w') as outf:
            create_datasets(outf, out_samples, subvol_size, num_nearest_images)

            index = 0
            for ndx in tqdm(range(len(f['x']))):
                # read everything once
                # these will change
                x, y, world_to_grid = f['x'][ndx], f['y'][ndx], f['world_to_grid'][ndx] 
                # these wont change
                scan_id, scene_id, frames = f['scan_id'][ndx], f['scene_id'][ndx], f['frames'][ndx]
                if (-1 not in f['frames'][ndx]) and ((f['x'][ndx]==1).sum()/(32*32*64) > 0.01): 

                    outf['scan_id'][index] = scan_id
                    outf['scene_id'][index] = scene_id
                    outf['x'][index] = x
                    outf['y'][index] = y
                    outf['world_to_grid'][index] = world_to_grid
                    outf['frames'][index] = frames
                    index +=1
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('in_path', help='Path to input h5 file')
    p.add_argument('out_path', help='Path to output h5 file')

    args = p.parse_args()

    main(args)