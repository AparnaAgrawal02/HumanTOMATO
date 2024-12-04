import os

path = "/scratch/aparna/BSL_t2m_test/"

out_path = "/scratch/aparna/BSL_t2m_test_ready/"
if not os.path.exists(out_path):
    os.makedirs(out_path)
os.makedirs(os.path.join(out_path, "smplx_322"), exist_ok=True)
os.makedirs(os.path.join(out_path, "new_joint_vecs"), exist_ok=True)
os.makedirs(os.path.join(out_path, "joint"), exist_ok=True)
os.makedirs(os.path.join(out_path, "new_joints"), exist_ok=True)

for gloss in os.listdir(path):
    if ".npy" in gloss:
        f1 = os.path.join(path, gloss)
        f2 = os.path.join(out_path, gloss)
        os.system(f"cp {f1} {f2}")
        continue
    for option in os.listdir(os.path.join(path, gloss)):

        f1 = os.path.join(path, gloss, option, 'smplx_322', 'smplx_322.npy') 
        f2 = os.path.join(out_path,"smplx_322" , gloss+'_'+option+".npy")
        os.system(f"cp {f1} {f2}")

        f1 = os.path.join(path, gloss, option, "new_joint_vecs", "smplx_322.npy")
        f2 = os.path.join(out_path,"new_joint_vecs" , gloss+'_'+option+".npy")

        os.system(f"cp {f1} {f2}")

        f1 = os.path.join(path, gloss, option, "joint", "smplx_322.npy")
        f2 = os.path.join(out_path,"joint" , gloss+'_'+option+".npy")
        os.system(f"cp {f1} {f2}")

        f1 = os.path.join(path, gloss, option, "new_joints", "smplx_322.npy")
        f2 =os.path.join(out_path,"new_joints" , gloss+'_'+option+".npy")
        os.system(f"cp {f1} {f2}")



        with open(os.path.join(out_path, "all.txt"), 'a+') as f:
            f.write(gloss + '_' + option + '\n')

        os.makedirs(os.path.join(out_path, "texts"), exist_ok=True)
        with open(os.path.join(out_path, "texts", gloss + '_' + option + '.txt'), 'w') as f:
            f.write(gloss + "." + '\n')