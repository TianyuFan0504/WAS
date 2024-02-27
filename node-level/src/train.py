import csv
import time
import json
import argparse
import warnings
import numpy as np
import torch
import torch.optim as optim

from utils import *
from models import *
from dataloader import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def teacher_model(set_of_ssl):

    model = Model(param).to(device)

    ssl_agent_list = []
    params = list(model.parameters())
    for _, ssl in enumerate(set_of_ssl):
        agent = eval(ssl)(g, model, param).to(device)
        ssl_agent_list.append(agent)
        if agent.disc is not None:
            params = params + list(agent.disc.parameters())

    optimizer = optim.Adam(params, lr=float(param["lr"]), weight_decay=float(param["weight_decay"]))
    criterion = torch.nn.NLLLoss()

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0
    out_best = None

    st = time.clock()
    for epoch in range(param['epoch']):
            
        model.train()
        out = model(g, feats)
        logits = out.log_softmax(dim=1)
        loss_nc = criterion(logits[idx_train], labels[idx_train])
        loss_ssl = 0
        for _, ssl in enumerate(ssl_agent_list):
            loss_ssl += ssl.make_loss(model.encoder(g, feats)[0][-2])
            loss = loss_ssl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        logits_eval = model(g, feats)
        train_acc = evaluation(logits_eval[idx_train], labels[idx_train])
        val_acc = evaluation(logits_eval[idx_val], labels[idx_val])
        test_acc = evaluation(logits_eval[idx_test], labels[idx_test])

        if test_acc > test_best:
            test_best = test_acc

        if val_acc > val_best:
            val_best = val_acc
            test_val = test_acc
            out_best = logits_eval
            es = 0
        else:
            es += 1
            if es == 200 and param['dataset'] != 'amazon-com' and param['dataset'] != 'ogbn-arxiv':
                print('Early stop!')
                break

        if epoch % 10 == 0:
            print('\033[0;30;41m Epoch [{:3}/{}]: NC Loss {:.6f}, SSL Loss {:.6f} | Train Acc {:.4f}, Val Acc {:.4f}, Test Acc {:.4f} | {:.4f}, {:.4f}\033[0m'.format(
                epoch+1, param['epoch'], loss_nc.item(), loss_ssl.item(), train_acc, val_acc, test_acc, test_val, test_best))

    

    optimizer = optim.Adam(model.parameters(), lr=float(param["lr"]), weight_decay=float(param["weight_decay"]))
    criterion = torch.nn.NLLLoss()

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0
    out_best = None

    for epoch in range(param['epoch']):
            
        model.train()
        out = model(g, feats)
        logits = out.log_softmax(dim=1)
        loss = criterion(logits[idx_train], labels[idx_train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        logits_eval = model(g, feats)
        train_acc = evaluation(logits_eval[idx_train], labels[idx_train])
        val_acc = evaluation(logits_eval[idx_val], labels[idx_val])
        test_acc = evaluation(logits_eval[idx_test], labels[idx_test])

        if test_acc > test_best:
            test_best = test_acc

        if val_acc > val_best:
            val_best = val_acc
            test_val = test_acc
            out_best = logits_eval
            es = 0
        else:
            es += 1
            if es == 200 and param['dataset'] != 'amazon-com' and param['dataset'] != 'ogbn-arxiv':
                print('Early stop!')
                break

        if epoch % 10 == 0:
            print('\033[0;30;43m Fine-tune Epoch [{:3}/{}]: NC Loss {:.6f} | Train Acc {:.4f}, Val Acc {:.4f}, Test Acc {:.4f} | {:.4f}, {:.4f}\033[0m'.format(
                epoch+1, param['epoch'], loss.item(), train_acc, val_acc, test_acc, test_val, test_best))

    et = time.clock()
    print('===========')
    print(set_of_ssl)
    print(et-st)
    print('===========')
    return out_best, test_acc, test_val, test_best
    

def student_model(logits_list):
    model = Model(param).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(param["lr"]), weight_decay=float(param["weight_decay"]))
    criterion_l = torch.nn.NLLLoss()
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0
    st = time.clock()
    for epoch in range(param['epoch']):
            
        model.train()
        # with torch.no_grad():
        #     model.momentum_update_key_encoder()
        out = model(g, feats)
        logits = out.log_softmax(dim=1)
        logits_target = model.estimate_optimal_distribution(out, logits_list, param)
        
        loss_nc = criterion_l(logits[idx_train], labels[idx_train])
        loss_fit = criterion_l(logits_target[idx_train], labels[idx_train])
        loss_kd = criterion_t(torch.log_softmax(out / param['tau'], dim=1), logits_target.detach())
        loss = loss_nc + loss_fit + param['beta'] * param['tau'] * param['tau'] * loss_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        logits_eval = model(g, feats)
        train_acc = evaluation(logits_eval[idx_train], labels[idx_train])
        val_acc = evaluation(logits_eval[idx_val], labels[idx_val])
        test_acc = evaluation(logits_eval[idx_test], labels[idx_test])

        if test_acc > test_best:
            test_best = test_acc

        if val_acc > val_best:
            val_best = val_acc
            test_val = test_acc
            es = 0
        else:
            es += 1
            if (es == 200 and param['dataset'] != 'ogbn-arxiv') or (es == 500 and param['dataset'] == 'ogbn-arxiv'):
                print('Early stop!')
                break

        if epoch % 10 == 0:
            print('\033[0;30;46m Epoch [{:3}/{}]: NC-S Loss {:.6f}, NC-T Loss {:.6f}, KD Loss {:.6f} | Train Acc {:.4f}, Val Acc {:.4f}, Test Acc {:.4f} | {:.4f}, {:.4f}\033[0m'.format(
                epoch+1, param['epoch'], loss_nc.item(), loss_fit.item(), loss_kd.item(), train_acc, val_acc, test_acc, test_val, test_best))
    et = time.clock()
    print('===========')
    print('student')
    print(et-st)
    print('===========')
    return test_acc, test_val, test_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--dataset", type=str, default="cora")

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_mode", type=int, default=0)
    parser.add_argument("--data_mode", type=int, default=0)
    parser.add_argument("--m", type=float, default=0.9)
    args = parser.parse_args()
    param = args.__dict__

    if param['data_mode'] == 0:
        param['dataset'] = 'cora'
    if param['data_mode'] == 1:
        param['dataset'] = 'citeseer'
    if param['data_mode'] == 2:
        param['dataset'] = 'pubmed'
    if param['data_mode'] == 3:
        param['dataset'] = 'coauthor-cs'
    if param['data_mode'] == 4:
        param['dataset'] = 'coauthor-phy'
    if param['data_mode'] == 5:
        param['dataset'] = 'amazon-photo'
    if param['data_mode'] == 6:
        param['dataset'] = 'amazon-com'
    if param['data_mode'] == 7:
        param['dataset'] = 'ogbn-arxiv'

    ssl_task_list = ['Par', 'Clu', 'DGI', 'PairwiseDistance', 'PairwiseAttrSim']
    param['num_teachers'] = len(ssl_task_list)
    g, feats, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    param['feat_dim'] = g.ndata["feat"].shape[1]
    param['label_dim'] = labels.int().max().item() + 1
    param['ssl_task_len'] = len(ssl_task_list)


    if param['save_mode'] == 0:
        set_seed(param['seed'])
        ssl_acc_list = []
        logits_list = []
        for ssl in ssl_task_list:
            path = '../models/{}/{}/{}_{}_{}_{}_{}'.format(param['dataset'], ssl, param['num_layers'], param['hidden_dim'], param['dropout'], param['alpha'], param['seed'])
            check_writable('../models/{}/{}'.format(param['dataset'], ssl), overwrite=False)
            if not os.path.exists(path + ".npz"):
                logits, test_acc, test_val, test_best = teacher_model([ssl])
                np.savez(path, logits.detach().cpu().numpy())
                if param['dataset'] != 'cora' and param['dataset'] != 'citeseer' and param['dataset'] != 'pubmed' and param['dataset'] != 'ogbn-arxiv':
                    np.save(path + '_train.npy', idx_train)
                    np.save(path + '_val.npy', idx_val)
                    np.save(path + '_test.npy', idx_test)
                print('saving {}/{}'.format(param['dataset'], ssl))
            else:
                logits = torch.from_numpy(np.load(path + ".npz")["arr_0"]).to(device)  
                if param['dataset'] != 'cora' and param['dataset'] != 'citeseer' and param['dataset'] != 'pubmed' and param['dataset'] != 'ogbn-arxiv':
                    idx_train = np.load(path + '_train.npy')
                    idx_val = np.load(path + '_val.npy')
                    idx_test = np.load(path + '_test.npy')
                print('loading {}/{}'.format(param['dataset'], ssl))
            ssl_acc_list.append(evaluation(logits[idx_test], labels[idx_test]))
            logits_list.append(logits)
        logits_list = torch.stack(logits_list, dim=0).detach()
        test_acc, test_val, test_best = student_model(logits_list)

    else:
        test_acc_list = []
        test_val_list = []
        test_best_list = []
        ssl_acc_list = []

        for seed in range(5):
            set_seed(seed + param['seed'])

            if seed == 0:
                logits_list = []
                for ssl in ssl_task_list:
                    path = '../models/{}/{}/{}_{}_{}_{}_{}'.format(param['dataset'], ssl, param['num_layers'], param['hidden_dim'], param['dropout'], param['alpha'], param['seed'])
                    check_writable('../models/{}/{}'.format(param['dataset'], ssl), overwrite=False)
                    if not os.path.exists(path + ".npz"):
                        logits, test_acc, test_val, test_best = teacher_model([ssl])
                        np.savez(path, logits.detach().cpu().numpy())
                        if param['dataset'] != 'cora' and param['dataset'] != 'citeseer' and param['dataset'] != 'pubmed' and param['dataset'] != 'ogbn-arxiv':
                            np.save(path + '_train.npy', idx_train)
                            np.save(path + '_val.npy', idx_val)
                            np.save(path + '_test.npy', idx_test)
                        print('saving {}/{}'.format(param['dataset'], ssl))
                    else:
                        logits = torch.from_numpy(np.load(path + ".npz")["arr_0"]).to(device)  
                        if param['dataset'] != 'cora' and param['dataset'] != 'citeseer' and param['dataset'] != 'pubmed' and param['dataset'] != 'ogbn-arxiv':
                            idx_train = np.load(path + '_train.npy')
                            idx_val = np.load(path + '_val.npy')
                            idx_test = np.load(path + '_test.npy')
                        print('loading {}/{}'.format(param['dataset'], ssl))
                    ssl_acc_list.append(evaluation(logits[idx_test], labels[idx_test]))
                    logits_list.append(logits)
                logits_list = torch.stack(logits_list, dim=0).detach()
            test_acc, test_val, test_best = student_model(logits_list)
            test_acc_list.append(test_acc)
            test_val_list.append(test_val)
            test_best_list.append(test_best)


    outFile = open('../PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in param.items():
        results.append(k)
    
    if param['save_mode'] == 0:
        results.append(str(ssl_acc_list))
        results.append(str(np.mean(ssl_acc_list)))
        results.append(str(test_acc))
        results.append(str(test_val))
        results.append(str(test_best))
    else:  
        results.append(str(ssl_acc_list))
        results.append(str(np.mean(ssl_acc_list)))
        results.append(str(test_val_list))
        results.append(str(np.mean(test_acc_list)))
        results.append(str(np.mean(test_val_list)))
        results.append(str(np.mean(test_best_list)))
        results.append(str(np.std(test_val_list)))
    writer.writerow(results)

