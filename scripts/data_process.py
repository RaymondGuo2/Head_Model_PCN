import os
import shutil
import random


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


# Going to be 80% train, 10% val, 10% test
def train_val_test_split(unsplit_directory_path, split_destination_path):
    unsplit_directory_path = os.path.expanduser(unsplit_directory_path)
    split_destination_path = os.path.expanduser(split_destination_path)

    os.makedirs(split_destination_path, exist_ok = True)
    unsplit_partial_data = os.path.join(unsplit_directory_path, 'partial')
    unsplit_gt_data = os.path.join(unsplit_directory_path, 'ground_truth')

    partial_inputs = [file for file in os.listdir(unsplit_partial_data)]
    matched_files = [(file, file.replace('.npy', '_gt.npy')) for file in partial_inputs if os.path.exists(os.path.join(unsplit_gt_data, file.replace('.npy', '_gt.npy')))]
    
    random.shuffle(matched_files)
    num_files = len(matched_files)
    train_range = int(num_files * 0.8)
    validation_range = train_range + int(num_files * 0.1)

    train_idx = matched_files[:train_range]
    val_idx = matched_files[train_range:validation_range]
    test_idx = matched_files[validation_range:]

    def assign_to_trainValTest(pairs, mode):
        partial_inputs = os.path.join(split_destination_path, mode, 'partial_inputs')
        ground_truths = os.path.join(split_destination_path, mode, 'ground_truths')

        for partial, gt in pairs:
            src_partial = os.path.join(unsplit_partial_data, partial)
            dest_partial = os.path.join(partial_inputs, partial)
            shutil.copy(src_partial, dest_partial)
            print(f"Copied {src_partial} to {dest_partial}")

            src_gt = os.path.join(unsplit_gt_data, gt)
            dest_gt = os.path.join(ground_truths, gt)
            shutil.copy(src_gt, dest_gt)
            print(f"Copied {src_gt} to {dest_gt}")

    assign_to_trainValTest(train_idx, 'train')
    assign_to_trainValTest(val_idx, 'val')
    assign_to_trainValTest(test_idx, 'test')



if __name__ == '__main__':
    unsplit_directory_path = '~/../../vol/bitbucket/rqg23/processed_pairs_data'
    split_destination_path = '~/../../vol/bitbucket/rqg23/project_data'
    train_val_test_split(unsplit_directory_path, split_destination_path)
