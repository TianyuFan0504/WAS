from operator import is_
from os.path import join
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import random
import argparse
from models import GNN, GNN_graphpred
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
from splitters import random_scaffold_split, random_split, scaffold_split
from torch_geometric.data import DataLoader

from prepare.datasets import MoleculeDataset
from tools import *

warnings.filterwarnings("ignore", category=Warning)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def train_teacher_model(set_of_ssl, batch):
    for ssl_task_name in set_of_ssl:
        teacher_model=param['model_dict'][ssl_task_name]
        test_roc, test_target, test_pred = eval_b(teacher_model, device, batch)
    return test_roc, test_target, test_pred

def train_student_model(model, optimizer, train_loader):
    device = torch.device('cuda:' + str(param['device'])) \
        if torch.cuda.is_available() else torch.device('cpu')
    criterion_l = torch.nn.BCELoss()
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0

    for epoch in range(1,param['epochs']+1):
            
        model.train()
        with torch.no_grad():
            model.momentum_update_key_encoder()
        for idx, batch in enumerate(train_loader):

            ssl_roc_list=[]
            logits_list=[]
            batch = batch.to(device)
            
            for ssl in ssl_task_list:
                test_roc, test_target, test_pred=train_teacher_model([ssl],batch)
                ssl_roc_list.append(test_roc)
                logits_list.append(test_pred)
            logits_list = torch.stack(logits_list, dim=0).detach()
            logits_list = logits_list.to(device)

            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            labels = batch.y.view(out.shape).float().detach()
            is_valid = labels ** 2 > 0
            logits_target, q, w = model.estimate_optimal_distribution(out, logits_list, param)

            loss_nc = criterion_l(out[is_valid].sigmoid(), (labels[is_valid]+1)/2)

            loss_fit = criterion_l(logits_target[is_valid].sigmoid(), (labels[is_valid]+1)/2)

            if param['num_tasks']==1:
                valid = torch.cat((is_valid,is_valid),dim = 1)
                out_k = torch.cat(((out / param['tau']).sigmoid(),(1-(out / param['tau']).sigmoid()).detach()),dim = 1)[valid]
                log_k = torch.cat(((logits_target / param['tau']).sigmoid(),(1-(logits_target / param['tau']).sigmoid()).detach()),dim = 1)[valid]
                loss_kd = criterion_t(out_k.log(), log_k.detach())
            else:
                loss_kd = criterion_t(torch.softmax(out / param['tau'],dim = 1)[is_valid].log(), torch.softmax(logits_target / param['tau'],dim = 1)[is_valid].detach())

            loss =  loss_nc +  param['alpha'] * loss_fit + param['beta'] * param['tau'] * param['tau'] * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        model.eval()

        train_roc    ,_  ,_ = eval_t(model, device, train_loader)
        val_roc    ,_  ,_ = eval_t(model, device, val_loader)
        test_roc   ,_  ,_ = eval_t(model, device, test_loader)

        if test_roc > test_best:
            test_best = test_roc

        if val_roc > val_best:
            val_best = val_roc
            test_val = test_roc
            es = 0
        else:
            es += 1
            if (es == 200 ):
                print('Early stop!')
                SaveModel(model, param)
                return 1, test_val, 1

        if epoch % 10 ==0 or epoch < 10:  
            print('\033[0;30;46m Epoch [{:3}/{}]: Loss {:.6f}, NC-S Loss {:.6f}, NC-T Loss {:.6f}, KD Loss {:.6f}| Train ROC {:.4f}, Val ROC {:.4f}, Test ROC {:.4f} | {:.4f}, {:.4f}\033[0m'.format(
                epoch, param['epochs'], loss.item(), loss_nc.item(), loss_fit.item(), loss_kd.item(), train_roc, val_roc, test_roc, test_val, test_best))

    return test_roc, test_val, test_best

def eval_b(model, device, batch):
    model.eval()
    y_true, y_scores = [], []
    batch = batch.to(device)
    with torch.no_grad():
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    true = batch.y.view(pred.shape)

    y_true.append(true)
    y_scores.append(pred)
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(eval_metric((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        else:
            print('{} is invalid'.format(i))


    if len(roc_list) < y_true.shape[1]:
        print(len(roc_list))
        print('Some target is missing!')
        print('Missing ratio: %f' %(1 - float(len(roc_list)) / y_true.shape[1]))
    assert not len(roc_list) == 0, "ERROR in Missing Ratio!"
    test_roc=sum(roc_list) / len(roc_list)      
    test_target=torch.tensor(y_true, dtype=float)
    test_pred=torch.tensor(y_scores)

    return test_roc, test_target,test_pred

def eval_t(model, device, loader):
    model.eval()
    y_true, y_scores = [], []

    for _,batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        true = batch.y.view(pred.shape)
        y_true.append(true)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            cal_y_scores=y_scores[is_valid, i]
            roc_list.append(eval_metric((y_true[is_valid, i] + 1) / 2, cal_y_scores))
        else:
            print('{} is invalid'.format(i))

    if len(roc_list) < y_true.shape[1]:
        print(len(roc_list))
        print('Some target is missing!')
        print('Missing ratio: %f' %(1 - float(len(roc_list)) / y_true.shape[1]))

    test_roc=sum(roc_list) / len(roc_list)     
    test_target=torch.tensor(y_true, dtype=float)
    test_pred=torch.tensor(y_scores)
    return test_roc, test_target,test_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # about seed and basic info
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runseed', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    # about dataset and dataloader
    parser.add_argument('--input_data_dir', type=str, default='./molecule_datasets/')
    parser.add_argument('--dataset', type=str, default='bace')
    parser.add_argument('--num_workers', type=int, default=8)
    # about training strategies
    parser.add_argument('--split', type=str, default='scaffold')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_scale', type=float, default=10)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.001)
    # about molecule GNN
    parser.add_argument('--gnn_type', type=str, default='gin')
    parser.add_argument('--num_layer', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--dropout_ratio', type=float, default=0.5)
    parser.add_argument('--graph_pooling', type=str, default='mean')
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument('--gnn_lr_scale', type=float, default=1)
    # about loading and saving
    parser.add_argument('--input_model_file', type=str, default='./teachers')
    parser.add_argument('--output_model_dir', type=str, default='./student')
    # about WAS
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--num_teachers", type=int, default=100)
    parser.add_argument("--model_dict", type=dict, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=0.9)

    args = parser.parse_args()
    param = args.__dict__

    setup_seed(param['runseed'])

    device = torch.device('cuda:' + str(param['device'])) \
        if torch.cuda.is_available() else torch.device('cpu')

    num_tasks = get_num_task(param['dataset'])
    param['num_tasks'] = num_tasks

    dataset_folder = './molecule_datasets/'

    root=dataset_folder + param['dataset']
    dataset = MoleculeDataset(root, dataset=param['dataset'])

    ssl_task_list = ['AM','EP', 'GPTGNN', 'CP', 'GraphCL', 'IG', 'GraphLoG']

    num_teachers = len(ssl_task_list)
    print('{} teachers, they are {}'.format(num_teachers,ssl_task_list))
    param['num_teachers'] = num_teachers

    eval_metric = roc_auc_score

    #split datasets
    if param['split'] == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + param['dataset'] + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.7,
            frac_valid=0.2, frac_test=0.1)
        print('split via scaffold')
    elif param['split'] == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=param['seed'])
        print('randomly split')
    elif param['split'] == 'random_scaffold':
        smiles_list = pd.read_csv(dataset_folder + param['dataset'] + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1, seed=param['seed'])
        print('random scaffold')
    else:
        raise ValueError('Invalid split option.')

    #set up loader
    train_loader = DataLoader(train_dataset, batch_size=param['batch_size'],
                              shuffle=True, num_workers=param['num_workers'])
    val_loader = DataLoader(valid_dataset, batch_size=param['batch_size'],
                            shuffle=True, num_workers=param['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=param['batch_size'],
                             shuffle=True, num_workers=param['num_workers'])
 
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    train_roc_list, val_roc_list, test_roc_list = [], [], []
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    best_val_roc, best_val_idx = -1, 0

    student, optimizer, scheduler = Load_Student_model(param)
    student=student.to(device)
    
    model_list=[]
    model_dict={}
    for ssl in ssl_task_list:
        model = Load_teacher_models(ssl, param)
        model_list.append(model)
    model_dict = dict(zip(ssl_task_list, model_list))
    param['model_dict'] = model_dict

    test_roc, test_val, test_best= train_student_model(student, optimizer, train_loader)

    print('test_val is ',test_val)
    if(test_roc != 1):
        SaveModel(model, param)

