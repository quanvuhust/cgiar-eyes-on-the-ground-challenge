import torch
import numpy as np
import time
from sklearn import metrics
import mlflow

def eval(val_loader, model, device, epoch, is_ema, default_configs):
    if is_ema:
        print("EMA EVAL")
    else:
        print("NORMAL EVAL")

    model.eval()
    predictions = []
    ground_truths = []
    scores = []

    val_metric = {"rmse": 10000}

    with torch.no_grad():
        for batch_idx, (imgs, extents, img_paths) in enumerate(val_loader):   
            imgs = imgs.to(device).float()
            extents = extents.to(device).float()*100

            extent_logits = model(imgs)
            preds = torch.nn.Sigmoid()(extent_logits)*100

            predictions += [preds]
            ground_truths += [extents.detach().cpu()]     

        predictions = torch.cat(predictions).cpu().numpy()

        ground_truths = torch.cat(ground_truths).cpu().numpy()
        rmse = metrics.mean_squared_error(ground_truths, predictions, squared=False)
        print('RMSE: ', rmse)

    val_metric['rmse'] = rmse 
         
    return val_metric, ground_truths, predictions