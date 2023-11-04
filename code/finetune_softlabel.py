import os
import mlflow

from timm.data.random_erasing import RandomErasing
from data_augmentations.mixup import mixup, cutmix

from torch.optim import Adam, SGD
from optimizer.sam import SAM
from optimizer.adan import Adan
from optimizer.ranger21.ranger21 import Ranger21
from optimizer.lion_pytorch.lion_pytorch import Lion

from datetime import datetime
import json

import pandas as pd
import numpy as np
import time
import torch
import random


from torch import nn
from torch.optim import lr_scheduler

import logging 

from timm.utils import ModelEma
from model import Net

from tqdm import tqdm

from utils import seed_torch, count_parameters, scheduler_lr, EarlyStopper
from datasets.dataset import ImageFolder
from torch.utils.data import DataLoader

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from eval import eval
import argparse
import gc

from timm.utils import get_state_dict

from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.distributed as dist

default_configs = {}

def remove_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def make_weight_folder(default_configs, fold):
    weight_path = os.path.join("weights", default_configs["a_name"])
    os.makedirs(weight_path, exist_ok=True)
    weight_path = os.path.join(weight_path, str(fold))
    os.makedirs(weight_path, exist_ok=True)
    return weight_path

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
        model.load_state_dict(new_pretrained_dict)
    return model

def build_criterion(default_configs, device_id):
    criterion_extent = nn.BCEWithLogitsLoss()
    criterion_extent.to(device_id)
    return criterion_extent

def build_net(default_configs, device_id):
    model = Net(default_configs["backbone"], device_id).to(device_id)
    
    return model

def build_optimizer(default_configs, model_without_ddp, device_id):
    num_tasks = dist.get_world_size()
    lr = default_configs["lr"]*num_tasks
    # lr = default_configs["lr"]
    if default_configs["optimizer"] == "SAM":
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer_model = SAM(model_without_ddp.parameters(), base_optimizer, lr=lr, momentum=0.9, weight_decay=default_configs["weight_decay"], adaptive=True)
    elif default_configs["optimizer"] == "Ranger21":
        optimizer_model = Ranger21(model_without_ddp.parameters(), lr=lr, weight_decay=default_configs["weight_decay"], 
        num_epochs=default_configs["num_epoch"], num_batches_per_epoch=len(train_loader))
    elif default_configs["optimizer"] == "SGD":
        optimizer_model = torch.optim.SGD(model_without_ddp.parameters(), lr=lr, weight_decay=default_configs["weight_decay"], momentum=0.9)
    elif default_configs["optimizer"] == "Lion":
        optimizer_model = Lion(model_without_ddp.parameters(), lr=lr, weight_decay=default_configs["weight_decay"])
    elif default_configs["optimizer"] == "Adan":
        optimizer_model = Adan(model_without_ddp.parameters(), lr=lr, weight_decay=default_configs["weight_decay"])
    
    return optimizer_model

def log_metric_mlflow(rank, metric_name, metric_value, step):
    if rank == 0:
        mlflow.log_metric(metric_name, metric_value, step=step)

def train_one_fold(fold, train_loader, test_loader, rank, num_tasks):
    early_stop = EarlyStopper()
    # Value to be scattered (rank 0 has the value)
    early_stop_flag_scatter = torch.tensor([1])   # Placeholder for other ranks
    scattered_values = [torch.zeros(1) for _ in range(dist.get_world_size())]
    random_erase = RandomErasing(probability=0.5)
    if rank == 0:
        print("FOLD: ", fold)
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    
    DATA_PATH = "train"
    start_epoch = 0

    scaler = torch.cuda.amp.GradScaler()
    weight_path = make_weight_folder(default_configs, fold)
    criterion_extent = build_criterion(default_configs, device_id)
    model = build_net(default_configs, device_id)
    old_weight_path = "weights/exp_16/0/checkpoint_11_ema.pt"
    print("Load old weight ", old_weight_path)
    model = load_old_weight(model, old_weight_path)
    
    if rank == 0:
        model_ema = ModelEma(
            model,
            decay=default_configs["model_ema_decay"],
            device=device_id, resume='')
    model_without_ddp = model    
    model = torch.compile(model)
    ddp_model = NativeDDP(model, device_ids=[device_id])
    
    optimizer_model = build_optimizer(default_configs, ddp_model, device_id)
    
    best_metric = {"rmse": 10000}
    best_metric_ema = {"rmse": {"score": 10000, "list": []}}
    best_model_path = ""

    input_list, output_list = [], []
    iter_size = 1

    for epoch in range(start_epoch, default_configs["num_epoch"]):
        if rank == 0:
            print("\n-----------------Epoch: " + str(epoch) + " -----------------")
        train_loader.sampler.set_epoch(epoch)

        # scheduler_lr(optimizer_model, epoch)
        for param_group in optimizer_model.param_groups:
            log_metric_mlflow(rank, "lr", param_group['lr'], step=epoch)
            
        start = time.time()
        optimizer_model.zero_grad()

        batch_idx = 0
        for imgs, labels, img_paths in tqdm(train_loader):
            ddp_model.train()
            imgs = imgs.to(device_id).float()
            labels = labels.to(device_id).float().view(-1, 1)

            if torch.rand(1)[0] < 0.5 and (default_configs["use_mixup"] or default_configs["use_cutmix"]):
                rand_prob = torch.rand(1)[0]

                if default_configs["use_mixup"] == True and default_configs["use_cutmix"] == False:
                    mix_images, target_a, target_b, lam = mixup(imgs, labels, alpha=default_configs["mixup_alpha"])
                if default_configs["use_mixup"] == False and default_configs["use_cutmix"] == True:
                    mix_images, target_a, target_b, lam = cutmix(imgs, labels, device_id, default_configs["mixup_alpha"])
                if default_configs["use_mixup"] == True and default_configs["use_cutmix"] == True: 
                    if rand_prob < 0.5:
                        mix_images, target_a, target_b, lam = mixup(imgs, labels, alpha=default_configs["mixup_alpha"])
                    else:
                        mix_images, target_a, target_b, lam = cutmix(imgs, labels, device_id, default_configs["mixup_alpha"])
                
                with torch.cuda.amp.autocast():
                    logits = ddp_model(mix_images)
                    loss = criterion_extent(logits, target_a) * lam + \
                    (1 - lam) * criterion_extent(logits, target_b)
                    loss /= default_configs["accumulation_steps"]
                
                scaler.scale(loss).backward()
                if ((batch_idx + 1) % default_configs["accumulation_steps"] == 0) or ((batch_idx + 1) == len(train_loader)):
                    # scaler.unscale_(optimizer_model)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), default_configs["max_norm"])
                    scaler.step(optimizer_model)
                    scaler.update()
                    if rank == 0:
                        model_ema.update(model_without_ddp)
                    optimizer_model.zero_grad()
            else:
                with torch.cuda.amp.autocast():
                    imgs = random_erase(imgs)
                    extent_logits = ddp_model(imgs)
                    loss = criterion_extent(extent_logits, labels)
                    loss /= default_configs["accumulation_steps"]
                scaler.scale(loss).backward()
                if ((batch_idx + 1) % default_configs["accumulation_steps"] == 0) or ((batch_idx + 1) == len(train_loader)):
                    # scaler.unscale_(optimizer_model)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), default_configs["max_norm"])
                    scaler.step(optimizer_model)
                    scaler.update()
                    if rank == 0:
                        model_ema.update(model_without_ddp)
                    optimizer_model.zero_grad()
        
            batch_idx += 1

        end = time.time()
        log_metric_mlflow(rank, "train_elapsed_time", end - start, step=epoch)
        if rank == 0:
            print("train elapsed time", end - start)
        dist.barrier()
        if rank == 0:
            val_metric_type_list = ["rmse"]
            start = time.time()
            val_metric, ground_truths, scores = eval(test_loader, model_ema.ema, device_id, epoch, True, default_configs)
            end = time.time()
            print("val elapsed time", end - start)
            for val_metric_type in val_metric_type_list:
                print("Val ema {}: {}".format(val_metric_type, val_metric[val_metric_type]))
                mlflow.log_metric("val_{}_ema".format(val_metric_type), val_metric[val_metric_type], step=epoch)
                flag = False
                if val_metric_type in ["loss", "rmse"]:
                    if(val_metric[val_metric_type] < best_metric_ema[val_metric_type]["score"]):
                        flag = True
                else:
                    if(val_metric[val_metric_type] > best_metric_ema[val_metric_type]["score"]):
                        flag = True 
                if flag == True:
                    best_model_path = os.path.join(weight_path, 'checkpoint_{}_best_ema.pt'.format(val_metric_type))
                    try:
                        os.remove(best_model_path)
                    except Exception as e:
                        print(e)
                    # exported_model = torch._dynamo.export(get_state_dict(model_ema), input)
                    exported_model = get_state_dict(model_ema)
                    torch.save(exported_model, best_model_path)
                    mlflow.log_artifact(best_model_path)
                    best_metric_ema[val_metric_type] = {"score": val_metric[val_metric_type], "list": [ground_truths, scores]}
                    # print("Save best model ema: ", best_model_path, val_metric[val_metric_type])
            if early_stop.early_stop(val_metric["rmse"]) == True:
                # Scatter the value from rank 0 to all other ranks
                dist.scatter(scattered_values, early_stop_flag_scatter, src=0)

        dist.barrier()
        if scattered_values[rank].item() == 1:
            break   

    del model
    del ddp_model
    torch.cuda.empty_cache()
    gc.collect()

    return best_metric_ema


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--exp", type=str, default="exp_1")
    args = parser.parse_args()
    dist.init_process_group("nccl")
 
    # experiment_id = mlflow.create_experiment(args.exp)
 
    rank = dist.get_rank()
    num_tasks = dist.get_world_size()
    print(f"Start running basic DDP example on rank {rank}.")
 
    existing_exp = mlflow.get_experiment_by_name(args.exp)
    if rank == 0:
        if not existing_exp:
            mlflow.create_experiment(args.exp)
 
    dist.barrier()
    experiment = mlflow.set_experiment(args.exp)
    experiment_id = experiment.experiment_id
 
    f = open(os.path.join('code/configs', "{}.json".format(args.exp)))
    default_configs = json.load(f)
    f.close()
    if rank == 0:
        print(default_configs)
 
    seed_torch()
    DATA_PATH = "train"
    avg_score = {"rmse": 0}

    fold = default_configs["fold"]

    train_df = pd.read_csv("code/data/softlabels_fold{}.csv".format(fold))
    val_df =  pd.read_csv("code/data/val_fold{}.csv".format(fold))
    train_data = ImageFolder(train_df, "datasets/train", default_configs, {default_configs["test_image_size"]: 9, int(default_configs["train_image_size"]): 7}, "train")
    test_data = ImageFolder(val_df, "datasets/train", default_configs, None, "val")
       
    sampler_train = torch.utils.data.DistributedSampler(
        train_data, num_replicas=num_tasks, rank=rank, shuffle=True, seed=2023,
    )

    print("Sampler_train = %s" % str(sampler_train))
    train_loader = DataLoader(train_data, sampler=sampler_train, batch_size=default_configs["batch_size"], pin_memory=True, 
        num_workers=default_configs["num_workers"], drop_last=True)

    test_loader = DataLoader(test_data, batch_size=int(default_configs["batch_size"]), 
            pin_memory=True, num_workers=default_configs["num_workers"], drop_last=False)
 
    if rank == 0:
        with mlflow.start_run(
            experiment_id=experiment_id,
        ) as parent_run:
            score = train_one_fold(fold, train_loader, test_loader, rank, num_tasks) 
            for k, v in avg_score.items():
                avg_score[k] += score[k]["score"]
                mlflow.log_metric("{}".format(k), score[k]["score"])
                # mlflow.log_metric("{}_onnx".format(k), onnx_metric[k])
                print("{}: ".format(k), score[k]["score"])
            mlflow.end_run()
    else:
        score = train_one_fold(fold, train_loader, test_loader, rank, num_tasks) 

