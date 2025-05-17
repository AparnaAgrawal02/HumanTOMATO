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
    print("Loading and smoothing smplx parameters from", root_path)
    smplx_params = {}
    frames = len(os.listdir(os.path.join(root_path,'smplx_optimized', 'smplx_params')))
    for frame_idx in range(1,frames+1):
        smplx_param_path = os.path.join(root_path, 'smplx_optimized', 'smplx_params', str(frame_idx) + '.json')
        print(smplx_param_path)
        with open(smplx_param_path) as f:
            smplx_param = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}
        
        smplx_params[frame_idx] = smplx_param
        keys = smplx_param.keys()

    # smooth smplx parameters
    for key in keys:
        if 'pose' in key:
            pose = np.stack([smplx_params[frame_idx][key].reshape(-1) for frame_idx in range(1,frames+1)])
            pose = smoothen_poses(pose, window_length=9)
            for i, frame_idx in enumerate(range(1,frames+1)):
                smplx_params[frame_idx][key] = pose[i]
                if key in ['body_pose', 'lhand_pose', 'rhand_pose']:
                    smplx_params[frame_idx][key] = smplx_params[frame_idx][key].reshape(-1,3)

                if key == 'body_pose':
                    #print(smplx_params[frame_idx][key].shape)
                    # for i in range(11):
                    smplx_params[frame_idx]['body_pose'][:11] = [0.0,0.0,0.0]
            
        else:
            
            item = np.stack([smplx_params[frame_idx][key] for frame_idx in range(1,frames+1)])
            item = savgol_filter(item, window_length=9, polyorder=2, axis=0)
            for i, frame_idx in enumerate(range(1,frames+1)):
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

    frame_number = len(data)
    print(frame_number)
    
    print(data.shape)

    fId = 0 # frame id of the mocap sequence
    pose_seq = []

    for fId in range(0, frame_number, down_sample):
        data_pose = data[fId]
        #pose_root = data_pose['root_pose'].reshape(1, 3)
        pose_root = np.zeros((1, 3))
    
    # # Apply rotation to all joints
        pose_root[0][0] = -np.pi
        #move 90 degree
        #pose_root[0,:] = [0, 0, 0]
        # make first 11 0 
       # data_pose['body_pose'][:11] = [0, 0, 0]
        pose_body = data_pose['body_pose'].reshape(1, 63)
        pose_hand = data_pose['lhand_pose'].reshape(1, 45)
        pose_hand = np.concatenate((pose_hand, data_pose['rhand_pose'].reshape(1, 45)), axis=1)
        pose_jaw = data_pose['jaw_pose'].reshape(1, 3)
        pose_expression = data_pose['expr'].reshape(1, 50)
        pose_face_shape = np.zeros((1, 100))
        pose_trans = data_pose['trans'].reshape(1, 3)
        #print(data_pose.keys())
        pose_body_shape = np.zeros((1,10))      #np.asarray([[0.421,-1.658,0.361,0.314,0.226,0.065,0.175,-0.150,-0.097,-0.191]])
        #print shapes
        print(pose_root.shape, pose_body.shape, pose_hand.shape, pose_jaw.shape, pose_expression.shape, pose_face_shape.shape, pose_trans.shape, pose_body_shape.shape)
        pose = np.concatenate((pose_root, pose_body, pose_hand, pose_jaw, pose_expression, pose_face_shape, pose_trans, pose_body_shape), axis=1)
        print(pose.shape)
        pose_seq.append(pose)

    pose_seq = np.concatenate(pose_seq, axis=0)
    

    return pose_seq


def process_pose_optimised(pose):
    pose_root = pose[:, :3]
    pose_root = compute_canonical_transform(torch.from_numpy(pose_root)).detach().cpu().numpy()
    pose[:, :3] = pose_root
    pose_trans = pose[:, 309:312]
    pose_trans = transform_translation(pose_trans)
    pose[:, 309:312] = pose_trans

    return pose

path = "/scratch/aparna/PHOENIX-2014-T/"#nearly/n_001_049_000_nearly/smplx_optimized/smplx_params"
output_path = "/scratch/aparna/GSL_t2m_optimised"
for gloss in os.listdir(path):
    gloss_path = pjoin(path, gloss)
    if not os.path.isdir(gloss_path):
        continue
    for option in os.listdir(gloss_path):
        option_path = pjoin(gloss_path, option, "smplx_optimized", "smplx_params")
        if not os.path.isdir(option_path) or os.path.exists(pjoin(output_path, gloss, option +".npy" )):
            continue
        #try:
        data = load_and_smooth(pjoin(gloss_path, option))
        #except:
            #print("Failed to load", gloss, option)
           # continue
        ex_fps = 25
        all_poses = []
        for i in range(1,len(data)+1):
            all_poses.append(data[i])
        all_poses = np.array(all_poses)
        pose_seq = get_smplx_322_optimised(all_poses, ex_fps)
        print(pose_seq.shape)

        pose_seq = process_pose_optimised(pose_seq)
        os.makedirs(pjoin(output_path, gloss), exist_ok=True)
        np.save(pjoin(output_path, gloss, option), pose_seq)