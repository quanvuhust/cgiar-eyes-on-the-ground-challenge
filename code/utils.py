import numpy as np
import time
import random

import os
"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""

import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

SEED = 2021

class EarlyStopper:
    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_metric = float('inf')

    def early_stop(self, validation_metric):
        if validation_metric < self.min_validation_metric:
            self.min_validation_metric = validation_metric
            self.counter = 0
        elif validation_metric > (self.min_validation_metric + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def seed_torch(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True

def count_parameters(module):
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in module.parameters()])))
 
def scheduler_lr(optimizer, ep):
    if ep <= 6:
        for p in optimizer.param_groups:
            p['lr'] = 0.000017
    else:
      for p in optimizer.param_groups:
            p['lr'] = 0.0000017


