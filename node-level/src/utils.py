import os
import dgl
import torch
import shutil
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


def evaluation(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(preds == labels)
    accuracy = correct.item() / len(labels)

    return accuracy
