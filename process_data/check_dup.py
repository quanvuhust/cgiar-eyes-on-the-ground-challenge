import json, cv2, numpy as np, itertools, random, pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from sklearn import model_selection
from copy import deepcopy
import cv2
import os
import pickle
from scipy.spatial import distance

feature_list = pickle.load(open('dino-all-feature-list.pickle','rb'))
filenames = pickle.load(open('dino-all-filenames.pickle','rb'))

feature_map = {}
for filename, feature in zip(filenames, feature_list):
    feature_map[filename] = feature

train_df = pd.read_csv("Train.csv")
filenames = []
values = []

c_dup = 0
for index, row in train_df.iterrows():
    if row['filename'] not in filenames:
        if "Copy.jpg" not in row['filename']:
            filenames.append(row['filename'])
            values.append((row['extent'], row['growth_stage'], row['damage']))
    else:
        c_dup += 1
        print("Duplicate: ", row['filename'])

print("Number duplicate file name: ", c_dup)
image_ids = {}
bbox_dup_check = {}
feature_dup_check = set()
index = 0
c_dup_bbox = 0
c_dup_feature = 0
c_remove = 0

clean_filenames = []
clean_growth_stages = []
clean_damages = []
clean_extents = []
clean_feature_list = []
for filename, value in zip(filenames, values):
    filepath = os.path.join("../imgs", filename)
    feature = feature_map[filepath]
    flag = False
    for old_feature, old_filename in zip(clean_feature_list, clean_filenames):
        if filename != old_filename:
            if(filename, old_filename) not in feature_dup_check:
                if distance.cosine(feature, old_feature) <= 0.02:
                    print("Duplicate feature: ", filename, old_filename, value, distance.cosine(feature, old_feature))
                    c_dup_feature += 1
                    feature_dup_check.add((filename, old_filename))
                    feature_dup_check.add((old_filename, filename))
                    flag = True
                    break
    if flag == True:
        continue
    
    clean_filenames.append(filename)
    clean_extents.append(value[0])
    clean_growth_stages.append(value[1])
    clean_damages.append(value[2])
    clean_feature_list.append(feature_map[filepath])

print("Total dup feature: ", c_dup_feature)
filenames = np.array(clean_filenames)
extents = np.array(clean_extents)
growth_stages = np.array(clean_growth_stages)
damages = np.array(clean_damages)

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
for fold, (train_index, test_index) in enumerate(skf.split(filenames, extents)):
    x_train = filenames[train_index]
    growth_stages_train = growth_stages[train_index]
    damages_train = damages[train_index]
    extent_train = extents[train_index]
    x_train = np.expand_dims(x_train, 1)
    extent_train = np.expand_dims(extent_train, 1)
    growth_stages_train = np.expand_dims(growth_stages_train, 1)
    damages_train = np.expand_dims(damages_train, 1)

    x_test = filenames[test_index]
    extent_test = extents[test_index]
    damages_test = damages[test_index]
    growth_stages_test = growth_stages[test_index]
    x_test = np.expand_dims(x_test, 1)
    extent_test = np.expand_dims(extent_test, 1)
    growth_stages_test = np.expand_dims(growth_stages_test, 1)
    damages_test = np.expand_dims(damages_test, 1)

    train_df = pd.DataFrame(np.concatenate((x_train, growth_stages_train,damages_train, extent_train), axis=1), columns=["filename", "growth_stages", "damages", "extent"])
    train_df.to_csv('code/data/train_fold{}.csv'.format(fold), index=False)
    test_df = pd.DataFrame(np.concatenate((x_test, growth_stages_test, damages_test, extent_test), axis=1), columns=["filename", "growth_stages", "damages", "extent"])
    test_df.to_csv('code/data/val_fold{}.csv'.format(fold), index=False)

filenames = np.expand_dims(filenames, 1)
growth_stages = np.expand_dims(growth_stages, 1)
damages = np.expand_dims(damages, 1)
extents = np.expand_dims(extents, 1)

train_df = pd.DataFrame(np.concatenate((filenames, growth_stages, damages, extents), axis=1), columns=["filename", "growth_stages", "damages", "extent"])
train_df.to_csv('code/data/train_all_data.csv', index=False)