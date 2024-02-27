from os.path import join
import os
from tkinter import E
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from models import GNN, GNN_graphpred
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
from splitters import random_scaffold_split, random_split, scaffold_split
from torch_geometric.data import DataLoader
# from util import get_num_task

from datasets import MoleculeDataset

def Load_teacher_models(ssl_task_name,param):
    device = torch.device('cuda:' + str(param['device'])) \
        if torch.cuda.is_available() else torch.device('cpu')
    # set up model
    num_tasks = get_num_task(param['dataset'])
    LoadPath = './teachers/'+param['dataset']+'/'+ssl_task_name+'_model.pth'

    dict=torch.load(LoadPath,map_location=device)
   
    molecule_model = GNN(num_layer=param['num_layer'], emb_dim=param['emb_dim'],
                         JK=param['JK'], drop_ratio=param['dropout_ratio'],
                         gnn_type=param['gnn_type'])

    molecule_model.load_state_dict(dict["molecule_model"])

    model = GNN_graphpred(param=param, num_tasks=num_tasks,
                          molecule_model=molecule_model)
    # model.from_pretrained(LoadPath)
    model.load_state_dict(dict['model'],strict=False)
    model.to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
    return model



def Load_Student_model(param):
    molecule_model = GNN(num_layer=param['num_layer'], emb_dim=param['emb_dim'],
                         JK=param['JK'], drop_ratio=param['dropout_ratio'],
                         gnn_type=param['gnn_type'])
    model = GNN_graphpred(param = param, num_tasks=get_num_task(param['dataset']),
                          molecule_model=molecule_model)

    model_param_group = [{'params': model.molecule_model.parameters()},
                         {'params': model.teacher_vector.parameters()},
                         {'params': model.global_vector.parameters()},
                         {'params': model.graph_pred_linear.parameters()},
                         {'params': model.projS.parameters()}]

    optimizer = optim.Adam(model_param_group, lr=param['lr'],
                           weight_decay=param['decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=param['step_size'], gamma=param['gamma'])
    return model, optimizer, scheduler


def SaveModel(model, param):
    output_model_path = param['output_model_dir']+'/'+'student_on_'+param['dataset']+'.pth'
    print(output_model_path)
    saved_model_dict = {
        'molecule_model': model.molecule_model.state_dict(),
        'model': model.state_dict()
    }
    torch.save(saved_model_dict, output_model_path)


def get_num_task(dataset):
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp', 'donor']:
        return 1
    elif dataset == 'pcba':
        return 92
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    raise ValueError('Invalid dataset name.')
