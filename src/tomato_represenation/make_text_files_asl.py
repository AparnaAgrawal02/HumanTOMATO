train_file = "/scratch/aparna/ASL_t2m/how2sign_realigned_train.csv"
val_file = "/scratch/aparna/ASL_t2m/how2sign_realigned_val.csv"
test_file = "/scratch/aparna/ASL_t2m/how2sign_realigned_test.csv"

path = "/scratch/aparna/ASL_t2m/"
#make all.txt, train.txt, val.txt, test.txt
#seperate textfile for each id

import pandas as pd
import os

# df = pd.read_csv(train_file, on_bad_lines='skip',sep='\t')
# print(df.head())
# df = df.dropna()
# ids = df["SENTENCE_NAME"]

# texts= df["SENTENCE"]



# put 10% of data in val.txt
# put 10% of data in test.txt
# put 80% of data in train.txt
df = pd.read_csv(train_file, on_bad_lines='skip',sep='\t')
ids = df["SENTENCE_NAME"]

texts= df["SENTENCE"]


aal_file ="/scratch/aparna/ASL_t2m/all.txt"
for id in range(len(ids)):
    if ids[id] +".pkl.npy" in os.listdir("/scratch/aparna/ASL_t2m/new_joint_vecs"):
        with open(aal_file, 'a+') as f:
            f.write(ids[id] + '\n')
        os.makedirs(os.path.join(path, "texts"), exist_ok=True)
        with open(os.path.join(path, "texts", ids[id] + '.txt'), 'w') as f:
            f.write(texts[id] + "." + '\n')

        with open(os.path.join(path, "train.txt"), 'a+') as f:
            f.write(ids[id] + '\n')
    
df = pd.read_csv(val_file, on_bad_lines='skip',sep='\t')
print(df.head())
df = df.dropna()
ids = df["SENTENCE_NAME"]

texts= df["SENTENCE"]

for id in range(len(ids)):
    if ids[id] +".pkl.npy" in os.listdir("/scratch/aparna/ASL_t2m/new_joint_vecs"):
        with open(aal_file, 'a+') as f:
            f.write(ids[id] + '\n')
        os.makedirs(os.path.join(path, "texts"), exist_ok=True)
        with open(os.path.join(path, "texts", ids[id] + '.txt'), 'w') as f:
            f.write(texts[id] + "." + '\n')

        with open(os.path.join(path, "val.txt"), 'a+') as f:
            f.write(ids[id] + '\n')

df = pd.read_csv(test_file, on_bad_lines='skip',sep='\t')
df = df.dropna()
ids = df["SENTENCE_NAME"]

texts= df["SENTENCE"]

for id in range(len(ids)):
    if ids[id] +".pkl.npy" in os.listdir("/scratch/aparna/ASL_t2m/new_joint_vecs"):
        with open(aal_file, 'a+') as f:
            f.write(ids[id] + '\n')
        os.makedirs(os.path.join(path, "texts"), exist_ok=True)
        with open(os.path.join(path, "texts", ids[id] + '.txt'), 'w') as f:
            f.write(texts[id] + "." + '\n')

        with open(os.path.join(path, "test.txt"), 'a+') as f:
            f.write(ids[id] + '\n')

aal_file ="/scratch/aparna/ASL_t2m/all.txt"
data =[]
with open(aal_file, 'r') as f:
    data = f.readlines()
data = [x.strip() for x in data]
print(len(data))   #30595
import random
random.shuffle(data)
print(data[:10])
val_data = data[:3059]
test_data = data[3059:6118]
train_data = data[6118:]
print(len(val_data), len(test_data), len(train_data))

with open(os.path.join(path, "val.txt"), 'w') as f:
    for id in val_data:
        f.write(id + '\n')

with open(os.path.join(path, "test.txt"), 'w') as f:

    for id in test_data:
        f.write(id + '\n')

with open(os.path.join(path, "train.txt"), 'w') as f:
    for id in train_data:
        f.write(id + '\n')
