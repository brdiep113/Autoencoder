import numpy as np
from utils.convert_ground_truth import *
import torch.nn.functional as F
import torch.nn as nn

def ocdnet_loss(x, y):
    score_weight = 0
    location_weight = 1
    descriptor_weight = 0
    return score_weight * score_loss(x, y) + location_weight * location_loss(x, y) +\
        descriptor_weight * descriptor_loss(x, y)


def score_loss(input, target):
    return


def location_loss(input, target):
    return F.mse_loss(input, target, reduction='sum')


def descriptor_loss(x, y):
    return
