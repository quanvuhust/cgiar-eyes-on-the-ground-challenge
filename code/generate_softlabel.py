import pandas as pd
import json
from datasets.dataset import ImageFolder
from torch.utils.data import DataLoader
from model import Net
import os
import torch
from tqdm.auto import tqdm
import numpy as np

def load_old_weight(model, weight_path):
    if weight_path is not None:
        pretrained_dict = torch.load(weight_path)

        model_dict = model.state_dict()
        new_pretrained_dict = {}
        # print(pretrained_dict.keys())
        # print(model_dict.keys())
        for k, v in pretrained_dict.items():
            k = k.replace("_orig_mod.", "").replace("backbone.", "")
            if k in model_dict and v.size() == model_dict[k].size():
                new_pretrained_dict[k] = v
            else:
                print("Don't load layer: ", k)
        model.load_state_dict(new_pretrained_dict, strict=True)
    return model

def build_net(default_configs, weights_path, device_id):
    model = Net(default_configs["backbone"], device_id).to(device_id)
    model = load_old_weight(model, weights_path)
    return model

if __name__ == '__main__':
    f = open(os.path.join('code/configs', "exp_16.json"))
    default_configs = json.load(f)
    default_configs["test_image_size"] = 384
    f.close()

    val_df = pd.read_csv("code/data/train_fold0.csv")
    test_data = ImageFolder(val_df, "datasets/train", default_configs, None, "softlabel")
    test_loader = DataLoader(test_data, batch_size=12, 
            pin_memory=True, num_workers=16, drop_last=False)
    device_id = "cuda:0"
    weight_paths = ["weights/exp_16/0/checkpoint_11_ema.pt"
    ]
    models = []
    for weight_path in weight_paths:
        model = build_net(default_configs, weight_path, device_id)
        model.eval()
        models.append(model)
    predictions = []
    ground_truths = []
    id_list = []
    extent_list = []
    n_images = 0
    with torch.no_grad():
        for batch_idx, (imgs, labels, ids) in enumerate(tqdm(test_loader)):   
            imgs = imgs.to(device_id).float()
            labels = labels.to(device_id).float()*100
            ensemble_extents = torch.zeros((imgs.shape[0], 1)).to(device_id)

            for model in models:
                logits = model(imgs)
                extents = torch.nn.Sigmoid()(logits)*100
                ensemble_extents += extents
            ensemble_extents /= len(models)
            predictions += [ensemble_extents]
            ground_truths += [labels]
            for j in range(len(ids)):
                id_list.append(ids[j])
                
        predictions = torch.cat(predictions).cpu().numpy()
        ground_truths = torch.cat(ground_truths).cpu().numpy()
        for i in range(predictions.shape[0]):
            extent = predictions[i][0]
            label = ground_truths[i]
            print(id_list[i], extent, label)
            soft_label = 0.3*extent + 0.7*label
            extent_list.append(soft_label)
id_list = np.array(id_list)
extent_list = np.array(extent_list)
id_list = np.expand_dims(id_list, 1)
extent_list = np.expand_dims(extent_list, 1)

df = pd.DataFrame(np.concatenate((id_list, extent_list), axis=1), columns=["filename", "extent"])
df.to_csv('code/data/softlabels_fold0.csv', index=False)