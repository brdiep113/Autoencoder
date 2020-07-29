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

def descriptor_loss(descriptor_pred, descriptor_target):

    loss = nn.BCEWithLogitsLoss()
    total_loss = 0
    dimensions = descriptor_target.shape
    r = dimensions[2]
    #print(r)
    c = dimensions[3]
    #print(c)
    #print(descriptor_target[1, :, 127, 127])
    #print(descriptor_pred[1, :, 127, 127])

    total_loss += loss(descriptor_pred[0, :, :, :], descriptor_target[0, :, :, :])

    return total_loss / r

def location_loss(location_pred, location_target):
    loss = nn.BCEWithLogitsLoss()
    return loss(location_pred, location_target)
