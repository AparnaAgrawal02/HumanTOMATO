# import os
# import numpy as np
# import json
# import cv2

# def load_json(path):
#     with open(path, 'r') as f:
#         return json.load(f)

# def load_image(path):
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     return img

# def process_folder(folder_path, output_path):
#     # Paths
#     cam_params_path = os.path.join(folder_path, 'cam_params')
#     flame_init_cam_path = os.path.join(folder_path, 'flame_init', 'cam')
#     flame_init_flame_params_path = os.path.join(folder_path, 'flame_init', 'flame_params')
#     keypoints_path = os.path.join(folder_path, 'keypoints_whole_body')
#     meta_path = os.path.join(folder_path, 'meta')
#     smplx_init_path = os.path.join(folder_path, 'smplx_init')
#     smplx_opt_params_path = os.path.join(folder_path, 'smplx_optimized', 'smplx_params')
#     frames_path = os.path.join(folder_path, 'frames')
#     if not os.path.exists(frames_path) or not os.path.exists(cam_params_path) or not os.path.exists(flame_init_cam_path) or not os.path.exists(flame_init_flame_params_path) or not os.path.exists(keypoints_path) or not os.path.exists(meta_path) or not os.path.exists(smplx_init_path) or not os.path.exists(smplx_opt_params_path):
#         print(f"Skipping {folder_path} as one of the required folders is missing.")
#         return

#     # Load all frames
#     frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.png')], key=lambda x: int(os.path.splitext(x)[0]))
#     frames = np.stack([load_image(os.path.join(frames_path, f)) for f in frame_files])

#     # Load all cam_params
#     cam_params = {f: load_json(os.path.join(cam_params_path, f)) for f in sorted(os.listdir(cam_params_path)) if f.endswith('.json')}
    
#     # Load flame_init
#     flame_cam = {f: load_json(os.path.join(flame_init_cam_path, f)) for f in sorted(os.listdir(flame_init_cam_path)) if f.endswith('.json')}
#     flame_params = {f: load_json(os.path.join(flame_init_flame_params_path, f)) for f in sorted(os.listdir(flame_init_flame_params_path)) if f.endswith('.json')}
    
#     # Load keypoints
#     keypoints = {f: load_json(os.path.join(keypoints_path, f)) for f in sorted(os.listdir(keypoints_path)) if f.endswith('.json')}
    
#     # Load meta
#     meta = {f: load_json(os.path.join(meta_path, f)) for f in sorted(os.listdir(meta_path)) if f.endswith('.json')}
    
#     # Load smplx_init
#     smplx_init = {f: load_json(os.path.join(smplx_init_path, f)) for f in sorted(os.listdir(smplx_init_path)) if f.endswith('.json')}
    
#     # Load smplx_optimized params
#     smplx_optimized = {f: load_json(os.path.join(smplx_opt_params_path, f)) for f in sorted(os.listdir(smplx_opt_params_path)) if f.endswith('.json')}

#     #add this ass well face_offset.json  joint_offset.json  locator_offset.json  shape_param.json  from smplx_optimised , aslo  shape_param.json from flame
#     # Pack into a dictionary
#     smplx_optimized_face_offset = load_json(os.path.join(os.path.join(folder_path, 'smplx_optimized'),"face_offset.json"))
#     smplx_optimised_joint_offset = load_json(os.path.join(os.path.join(folder_path, 'smplx_optimized'),"joint_offset.json"))
#     smplx_optimised_shape_param = load_json(os.path.join(os.path.join(folder_path, 'smplx_optimized'),"shape_param.json"))
#     smplx_optimised_locator_offset = load_json(os.path.join(os.path.join(folder_path, 'smplx_optimized'),"locator_offset.json"))
#     flame_shape_param = load_json(os.path.join(os.path.join(folder_path, 'flame_init'),"shape_param.json"))
#     data = {
#         "frames": frames,  # numpy array [N, H, W, 3]
#         "cam_params": cam_params,
#         "flame_init_cam": flame_cam,
#         "flame_init_flame_params": flame_params,
#         "keypoints": keypoints,
#         "meta": meta,
#         "smplx_init": smplx_init,
#         "smplx_optimized": smplx_optimized,
#         "smplx_optimized_face_offset": smplx_optimized_face_offset,
#         "smplx_optimised_joint_offset": smplx_optimised_joint_offset,
#         "smplx_optimised_shape_param": smplx_optimised_shape_param,
#         "smplx_optimised_locator_offset": smplx_optimised_locator_offset,
#         "flame_shape_param": flame_shape_param,
#     }

#     # Save as .npz compressed
#     np.savez_compressed(output_path, **data)
#     print(f"Saved: {output_path}")

# # Example usage:
# parent_dir = '/scratch/aparna/PHOENIX-2014-T'
# for split in os.listdir(parent_dir):
#     for folder in os.listdir(os.path.join(parent_dir, split)):
#         full_path = os.path.join(parent_dir, split, folder)
#         output_file = os.path.join(parent_dir+"_compressed",split ,folder + '.npz')
#         os.makedirs(os.path.dirname(output_file), exist_ok=True)
#         if os.path.isdir(full_path):
#             process_folder(full_path, output_file)
 
