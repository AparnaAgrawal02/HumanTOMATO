import numpy as np
import os
from os.path import join as pjoin

def mean_variance(data_dir, save_dir, joints_num):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        print(f"Loading {file}")
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(f"⚠️ Skipping file with NaNs: {file}")
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)  # (N, D)
    print(f"Total concatenated data shape: {data.shape}")

    # Now handle structured slicing
    pose = data

    slices = {
        'root_orient': (0, 3),
        'pose_body': (3, 66),
        'pose_hand': (66, 156),
        'pose_jaw': (156, 159),
        'face_expr': (159, 209),
        'face_shape': (209, 309),
        'trans': (309, 312),
        'betas': (312, pose.shape[-1])
    }

    Mean_parts = []
    Std_parts = []

    for part, (start, end) in slices.items():
        part_data = pose[:, start:end]
        part_mean = part_data.mean(axis=0)
        part_std = part_data.std(axis=0)

        # Normalize inside each block
        part_std[:] = part_std.mean()

        Mean_parts.append(part_mean)
        Std_parts.append(part_std)

        print(f"{part}: shape ({end-start}), mean {part_mean.shape}, std {part_std.shape}")

    # Stack all together
    Mean = np.concatenate(Mean_parts, axis=0)
    Std = np.concatenate(Std_parts, axis=0)

    print(f"Final Mean shape: {Mean.shape}, Final Std shape: {Std.shape}")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std

# Example usage
data_dir = "/scratch/aparna/datasets/GSL_t2m/smplx_322_optimised"
save_dir = "/scratch/aparna/datasets/GSL_t2m/smplx_322_mean_std"
mean_variance(data_dir, save_dir, 52)
