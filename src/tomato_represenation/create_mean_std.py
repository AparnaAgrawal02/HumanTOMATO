import numpy as np
import sys
import os
from os.path import join as pjoin


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def mean_variance(data_dir, save_dir, joints_num):
    
    file_list =[]
    for gloss in os.listdir(data_dir):
        if ".npy" in gloss:
            continue
        for dir1 in os.listdir(pjoin(data_dir, gloss)):
            file = pjoin(data_dir, gloss, dir1, "new_joint_vecs","smplx_322.npy")
            file_list.append(file)
    # file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        print(file)
        data = np.load( file)
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    print(Mean.shape, Std.shape)
    Std[0:1] = Std[0:1].mean() / 1.0  #1
    Std[1:3] = Std[1:3].mean() / 1.0  #2
    Std[3:4] = Std[3:4].mean() / 1.0  #1
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0    #3
    # Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0    #6  
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0

    print(Std.shape,8 + (joints_num - 1) * 9 + joints_num * 3,"fck")
    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std

data_dir = "/scratch/aparna/BSL_t2m_test/"
save_dir = "/scratch/aparna/BSL_t2m_test/"
mean_variance(data_dir, save_dir, 52)