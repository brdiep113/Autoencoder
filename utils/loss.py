import numpy as np
from utils.convert_ground_truth import *
import torch.nn.functional as F
import torch.nn as nn
import torch


def ocdnet_loss(score_pred, score_target, location_pred, location_target, descriptor_pred, descriptor_target):
    score_weight = 1
    location_weight = 1
    descriptor_weight = 1
    return score_weight * score_loss(score_pred, score_target) + \
           location_weight * location_loss(location_pred, location_target) +\
           descriptor_weight * descriptor_loss(descriptor_pred, descriptor_target)


def score_loss(score_pred, score_target):
    return 0

def descriptor_loss(location_target, descriptor_pred, descriptor_target):


    total_loss = 0
    dimensions = descriptor_target.shape
    r = dimensions[1]
    c = dimensions[2]
    for i in range(r):
        for j in range(c):
            total_loss += nn.BCEWithLogitsLoss(descriptor_pred[:, r, c], descriptor_target[:, r, c])

    return total_loss / r

def location_loss(location_pred, location_target):
    return nn.BCEWithLogitsLoss(location_pred, location_target)
