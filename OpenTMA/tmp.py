import numpy as np

path = "/scratch/aparna/experiments/temos/BSL/embeddings/val/epoch_3099/motion_embedding.npy"
path2 = "/scratch/aparna/experiments/temos/BSL/embeddings/val/epoch_3099/test_name_list.txt"

list1 = np.loadtxt(path2, dtype=str)
print(list1)

motion_embedding = np.load(path)
print(motion_embedding.shape)

# find the nearest neighbor of 0 index motion
distances = np.linalg.norm(motion_embedding - motion_embedding[554], axis=1)
print(distances, len(distances))
print(list1[554],"original")
# find index and the distance of the nearest 4 neighbor
print(np.argsort(distances))
print(np.sort(distances))
#print the text of the nearest 4 neighbor
print(list1[np.argsort(distances)[:4]])
# print(motion_embedding[3688])
# print(motion_embedding[0])
