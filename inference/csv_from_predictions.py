import os
import argparse
from feature_extraction import extract_features
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count

try:
    import ray
except ImportError:
    ray = None


def main(args):
    data_list = []
    files = os.listdir(args.input_dir)
    files = [f for f in files if f.endswith('.nii.gz') or f.endswith('.nii')]

    for f in tqdm(files):
        try:
            if f.endswith('.nii.gz'):
                path = os.path.join(args.input_dir, f)
                sub_id = f.split('_')[0]
                out = extract_features(path)
                out['sub_id'] = sub_id
                data_list.append(out)
        except Exception as e:
            print(f"Error processing file {f}: {str(e)}")
            continue

    if not data_list:
        print("No valid data was processed. Check the input files and error messages above.")
        return

    df = out_list_to_df(data_list)
    df.to_csv(args.output_path, index=False)

def ray_main(args):
    ray.init(num_cpus=(cpu_count()-1))
    data_list = []

    remote = ray.remote(extract_features)
    files = os.listdir(args.input_dir)
    files = [f for f in files if f.endswith('.nii.gz') or f.endswith('.nii')]
    futures = []
    for f in files:
        if f.endswith('.nii.gz'):
            path = os.path.join(args.input_dir, f)
            sub_id = f.split('_')[0]
            future = remote.remote(path)
            futures.append((future, sub_id))

    for future, sub_id in futures:
        try:
            out = ray.get(future)
            out['sub_id'] = sub_id
            data_list.append(out)
        except Exception as e:
            print(f"Error processing subject {sub_id}: {str(e)}")
            continue

    if not data_list:
        print("No valid data was processed. Check the input files and error messages above.")
        return

    df = out_list_to_df(data_list)
    df.to_csv(args.output_path, index=False)

def out_list_to_df(out_list):
    df_list = []
    for item in out_list:
        try:
            out = {}
            out['sub_id'] = item['sub_id']
            out['num_branches'] = item['num_branches']
            out['total_volume'] = item['total_volume']
            out['bifurcations'] = float(item['bifurcations'].sum())
            out['endpoints'] = float(item['endpoints'].sum())
            out['radius_list'] = item['radius_list']

            # Handle potential empty radius list
            if item['radius_list']:
                out['mean_radius'] = float(sum(item['radius_list']) / len(item['radius_list']))
                out['max_radius'] = float(max(item['radius_list']))
                out['min_radius'] = float(min(item['radius_list']))
            else:
                out['mean_radius'] = 0.0
                out['max_radius'] = 0.0
                out['min_radius'] = 0.0

            # Calculate tortuosities
            tortiousities = np.array([branch['tortuosity'] for branch in item['branch_list']])
            out['mean_tortuosity'] = float(np.mean(tortiousities))
            out['max_tortuosity'] = float(np.max(tortiousities))
            out['min_tortuosity'] = float(np.min(tortiousities))
            out['tortuosity_list'] = tortiousities.tolist()

            # Calculate branch lengths
            branch_lengths = [float(branch['full_path']) for branch in item['branch_list']]
            out['branch_list'] = [{'full_path': float(branch['full_path']),
                                   'straight_path': float(branch['straight_path']),
                                   'tortuosity': float(branch['tortuosity'])} for branch in item['branch_list']]

            if branch_lengths:
                out['total_branch_length'] = float(np.sum(branch_lengths))
                out['mean_branch_length'] = float(np.mean(branch_lengths))
                out['max_branch_length'] = float(np.max(branch_lengths))
            else:
                out['total_branch_length'] = 0.0
                out['mean_branch_length'] = 0.0
                out['max_branch_length'] = 0.0

            df_list.append(out)
        except Exception as e:
            print(f"Error processing data for subject {item.get('sub_id', 'unknown')}: {str(e)}")
            continue

    if not df_list:
        print("Warning: No valid data was processed into the DataFrame.")
        return pd.DataFrame()

    return pd.DataFrame(df_list)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_dir", type=str, required=True)
    args.add_argument("--output_path", type=str, required=True)
    args.add_argument("--no_ray", type=bool, default=False)

    args = args.parse_args()

    if ray is not None and not args.no_ray:
        ray_main(args)
    else:
        main(args)
