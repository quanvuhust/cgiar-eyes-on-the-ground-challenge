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


train_df = pd.read_csv("Train.csv")
filenames = []
growth_stages = []
damages = []
extents = []
groups = []
prefixes = {}
i = 0
for index, row in train_df.iterrows():
    if "Copy.jpg" not in row['filename']:
        if row['filename'][0] == 'L':
            pre = row['filename'][:-9]
            # print(pre)
            if pre not in prefixes.keys():
                prefixes[pre] = i
                i += 1
            filenames.append(row['filename'])
            groups.append(prefixes[pre])
            growth_stages.append(row['growth_stage'])
            damages.append(row['damage'])
            extents.append(row['extent'])
        elif 'repeat' in row['filename']:
            splits = row['filename'].split('_')
            pre = '_'.join(splits[:4])
            if pre not in prefixes.keys():
                prefixes[pre] = i
                i += 1
            filenames.append(row['filename'])
            groups.append(prefixes[pre])
            growth_stages.append(row['growth_stage'])
            damages.append(row['damage'])
            extents.append(row['extent'])
        else:
            splits = row['filename'].split('_')
            pre = '_'.join(splits[:3])
            if pre not in prefixes.keys():
                prefixes[pre] = i
                i += 1
            filenames.append(row['filename'])
            groups.append(prefixes[pre])
            growth_stages.append(row['growth_stage'])
            damages.append(row['damage'])
            extents.append(row['extent'])
print(len(filenames))
print(len(prefixes.keys()))

filenames = np.array(filenames)
extents = np.array(extents)
growth_stages = np.array(growth_stages)
damages = np.array(damages)
groups = np.array(groups)

from sklearn.model_selection import GroupKFold

skf = GroupKFold(n_splits=5)
for fold, (train_index, test_index) in enumerate(skf.split(filenames, extents, groups)):
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

    train_df = pd.DataFrame(np.concatenate((x_train, growth_stages_train,damages_train, extent_train), axis=1), columns=["filename", "growth_stage", "damage", "extent"])
    train_df.to_csv('../code/data/train_fold{}.csv'.format(fold), index=False)
    test_df = pd.DataFrame(np.concatenate((x_test, growth_stages_test, damages_test, extent_test), axis=1), columns=["filename", "growth_stage", "damage", "extent"])
    test_df.to_csv('../code/data/val_fold{}.csv'.format(fold), index=False)

filenames = np.expand_dims(filenames, 1)
growth_stages = np.expand_dims(growth_stages, 1)
damages = np.expand_dims(damages, 1)
extents = np.expand_dims(extents, 1)

train_df = pd.DataFrame(np.concatenate((filenames, growth_stages, damages, extents), axis=1), columns=["filename", "growth_stage", "damage", "extent"])
train_df.to_csv('../code/data/train_all_data.csv', index=False)