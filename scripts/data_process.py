import os
import shutil


def filter_for_npy(source_partial_dir_path, target_partial_dir_path, source_gt_dir_path, target_gt_dir_path):

    source_partial_dir_path = os.path.expanduser(source_partial_dir_path)
    target_partial_dir_path = os.path.expanduser(target_partial_dir_path)
    source_gt_dir_path = os.path.expanduser(source_gt_dir_path)
    target_gt_dir_path = os.path.expanduser(target_gt_dir_path)

    os.makedirs(target_partial_dir_path, exist_ok=True)
    os.makedirs(target_gt_dir_path, exist_ok=True)

    partial_list = []
    for file in os.listdir(source_partial_dir_path):
        if file.endswith('.npy'):
            partial_list.append(file)
            src = os.path.join(source_partial_dir_path, file)
            dest = os.path.join(target_partial_dir_path, file)

            shutil.copy(src, dest)
            print(f"Copied {src} to {dest}")    

    for file in os.listdir(source_gt_dir_path):
        partial_file = file.replace('_gt', '')
        if file.endswith('.npy') and partial_file in partial_list:
            src = os.path.join(source_gt_dir_path, file)
            dest = os.path.join(target_gt_dir_path, file)

            shutil.copy(src, dest)
            print(f"Copied {src} to {dest}")  


if __name__ == '__main__':
    source_partial_dir_path = '~/../../vol/bitbucket/rqg23/preprocessed_obj_partial_inputs'
    target_partial_dir_path = '~/../../vol/bitbucket/rqg23/processed_pairs_data/partial'
    source_gt_dir_path = '~/../../vol/bitbucket/rqg23/ground_truth_obj_processed'
    target_gt_dir_path = '~/../../vol/bitbucket/rqg23/processed_pairs_data/ground_truth'
    filter_for_npy(source_partial_dir_path, target_partial_dir_path, source_gt_dir_path, target_gt_dir_path)

