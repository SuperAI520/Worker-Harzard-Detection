import pandas as pd
from glob import glob
import random
import os 

train_rate=0.9
filenames = glob("./dataset/images/*")
random.shuffle(filenames)
print(len(filenames))

num_train = int(len(filenames)*train_rate)
train_filenames = filenames[:num_train]
val_filenames = filenames[num_train:]

with open("./data/train.txt", "w+") as f:
    for name in train_filenames:
        # name = os.path.basename(name).replace("txt", "jpg")
        f.write(name+"\n")
with open("./data/val.txt", "w+") as f:
    for name in val_filenames:
        # name = os.path.basename(name).replace("txt", "jpg")
        f.write(name+"\n")