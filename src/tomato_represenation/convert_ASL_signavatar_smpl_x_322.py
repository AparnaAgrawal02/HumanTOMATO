# import codecs as cs
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
import math
import torch
#from utils.rotation_conversions import *
import copy
#from utils.face_z_align_util import joint_idx, face_z_transform
from smplx import SMPLX
from utils.geometry import *
import json
import os
from scipy.signal import savgol_filter
def rotate_motion(root_global_orient):
    trans_matrix = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    motion = np.dot(root_global_orient, trans_matrix)  # exchange the y and z axis

    return motion

def compute_canonical_transform(global_orient):
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=global_orient.dtype)
    global_orient_matrix = axis_angle_to_matrix(global_orient)
    global_orient_matrix = torch.matmul(rotation_matrix, global_orient_matrix)
    global_orient = matrix_to_axis_angle(global_orient_matrix)
    return global_orient

def transform_translation(trans):
    trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
    trans[:, 2] = trans[:, 2] * (-1)
    return trans


# dict_keys(['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans'])
# root_pose (3,)
# body_pose (21, 3)
# jaw_pose (3,)
# leye_pose (3,)
# reye_pose (3,)
# lhand_pose (15, 3)
# rhand_pose (15, 3)
# expr (50,)
# trans (3,) 
def fix_quaternions(quats):
    """
    From https://github.com/facebookresearch/QuaterNet/blob/ce2d8016f749d265da9880a8dcb20a9be1a6d69c/common/quaternion.py

    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    :param quats: A numpy array of shape (F, N, 4).
    :return: A numpy array of the same shape.
    """
    assert len(quats.shape) == 3
    assert quats.shape[-1] == 4

    result = quats.copy()
    dot_products = np.sum(quats[1:] * quats[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result

def smoothen_poses(poses, window_length):
    """Smooth joint angles. Poses and global_root_orient should be given as rotation vectors."""
    n_joints = poses.shape[1] // 3

    # Convert poses to quaternions.
    qs = matrix_to_quaternion(axis_angle_to_matrix(torch.FloatTensor(poses).view(-1,3))).numpy()
    qs = qs.reshape((-1, n_joints, 4))
    qs = fix_quaternions(qs)

    # Smooth the quaternions.
    qs_smooth = []
    for j in range(n_joints):
        qss = savgol_filter(qs[:, j], window_length=window_length, polyorder=2, axis=0)
        qs_smooth.append(qss[:, np.newaxis])
    qs_clean = np.concatenate(qs_smooth, axis=1)
    qs_clean = qs_clean / np.linalg.norm(qs_clean, axis=-1, keepdims=True)

    ps_clean = matrix_to_axis_angle(quaternion_to_matrix(torch.FloatTensor(qs_clean).view(-1,4))).numpy()
    ps_clean = np.reshape(ps_clean, [-1, n_joints * 3])
    return ps_clean


def load_and_smooth(root_path):
    smplx_params = {}
    frames = len(os.listdir(os.path.join(root_path,'smplx_optimized', 'smplx_params')))
    for frame_idx in range(frames):
        smplx_param_path = os.path.join(root_path, 'smplx_optimized', 'smplx_params', str(frame_idx) + '.json')
        with open(smplx_param_path) as f:
            smplx_param = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}
        
        smplx_params[frame_idx] = smplx_param
        keys = smplx_param.keys()

    # smooth smplx parameters
    for key in keys:
        if 'pose' in key:
            pose = np.stack([smplx_params[frame_idx][key].reshape(-1) for frame_idx in range(frames)])
            pose = smoothen_poses(pose, window_length=9)
            for i, frame_idx in enumerate(range(frames)):
                smplx_params[frame_idx][key] = pose[i]
                if key in ['body_pose', 'lhand_pose', 'rhand_pose']:
                    smplx_params[frame_idx][key] = smplx_params[frame_idx][key].reshape(-1,3)

                if key == 'body_pose':
                    #print(smplx_params[frame_idx][key].shape)
                    # for i in range(11):
                    smplx_params[frame_idx]['body_pose'][:11] = [0.0,0.0,0.0]
            
        else:
            
            item = np.stack([smplx_params[frame_idx][key] for frame_idx in range(frames)])
            item = savgol_filter(item, window_length=9, polyorder=2, axis=0)
            for i, frame_idx in enumerate(range(frames)):
                smplx_params[frame_idx][key] = item[i]
    return smplx_params

def get_smplx_322_optimised(data, ex_fps):
    fps = 0


    if 'mocap_frame_rate' in data:
        fps = data['mocap_frame_rate']
        print(fps)
        down_sample = int(fps / ex_fps)
        
    elif 'mocap_framerate' in data:
        fps = data['mocap_framerate']
        print(fps)
        down_sample = int(fps / ex_fps)
    else:
        # down_sample = 1
        fps = 25
        down_sample = int(fps / ex_fps)

    #frame_number = len(data)
    #print(frame_number)
    frame_number = len(data['smplx'])
    #print(data.shape)
    pose_seq = []


    all_parameters = data['smplx']
    root_pose, body_pose, left_hand_pose, right_hand_pose, jaw_pose, shape, expression, cam_trans = \
    all_parameters[:, :3], all_parameters[:, 3:66], all_parameters[:, 66:111], all_parameters[:, 111:156], \
    all_parameters[:, 156:159], all_parameters[:, 159:169], all_parameters[:, 169:179], all_parameters[:, 179:182]
    cam_trans = np.zeros_like(all_parameters[:, 179:182])
    pose_face_shape = np.zeros((1, 100))
    pose_body_shape = np.asarray([[0.421,-1.658,0.361,0.314,0.226,0.065,0.175,-0.150,-0.097,-0.191]])
    #repeat the face shape and body shape for all frames
    pose_face_shape = np.repeat(pose_face_shape, frame_number, axis=0)
    pose_body_shape = np.repeat(pose_body_shape, frame_number, axis=0)

    #convert expressions to 50 dim
    pose_expression = np.zeros((frame_number, 50))
    pose_expression[:, :10] = expression[:, :10]
    print(root_pose.shape, body_pose.shape, left_hand_pose.shape, right_hand_pose.shape, jaw_pose.shape, pose_expression.shape, pose_face_shape.shape)
    pose_seq = np.concatenate((root_pose, body_pose, left_hand_pose, right_hand_pose, jaw_pose, pose_expression, pose_face_shape, cam_trans, pose_body_shape), axis=1)   
    print(pose_seq.shape)
    

    return pose_seq


def process_pose_optimised(pose):
    pose_root = pose[:, :3]
    pose_root = compute_canonical_transform(torch.from_numpy(pose_root)).detach().cpu().numpy()
    pose[:, :3] = pose_root
    pose_trans = pose[:, 309:312]
    pose_trans = transform_translation(pose_trans)
    pose[:, 309:312] = pose_trans

    return pose

path = "/scratch/aparna/how2sign_pkls_cropTrue_shapeTrue/"#nearly/n_001_049_000_nearly/smplx_optimized/smplx_params"
output_path = "/scratch/aparna/ASL_t2m"
for file in os.listdir(path):
        all_poses = np.load(pjoin(path, file), allow_pickle=True)
        #print(all_poses.shape)
        pose_seq = get_smplx_322_optimised(all_poses, 24)
        print(pose_seq.shape)

        pose_seq = process_pose_optimised(pose_seq)
        os.makedirs(pjoin(output_path , "smplx_322"), exist_ok=True)
        np.save(pjoin(output_path, "smplx_322", file), pose_seq)