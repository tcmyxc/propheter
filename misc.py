import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import torch
import os
import json

import loaders

import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from torch import optim
from functools import partial
from sklearn.metrics import classification_report, accuracy_score

from loss.fl import focal_loss
from loss.bsl import balanced_softmax_loss
from loss.cbl import CB_loss


def evaluate(dataloader, model, device, args):
    y_pred_list = []
    y_true_list = []
    
    size = len(dataloader.dataset)
    model.eval()
        
    test_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        y_true_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)
        pred = model(X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        y_pred_list.extend(pred.argmax(1).cpu().numpy())

    correct /= size
        
    print(f"\n[INFO] Test Error: Accuracy: {(100*correct):>0.2f}%\n")
    
    if args.print_report:
        print(classification_report(y_true_list, y_pred_list, digits=4))
    
    return correct


def train_one_epoch(dataloader, model, loss_fn, optimizer, device, print_report=True, print_freq=10):
    y_pred_list = []
    y_true_list = []

    train_loss, correct = 0, 0
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        y_true_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(True):
            pred = model(X)  # forward

            loss = loss_fn(pred, y)
            train_loss += loss.item()

            y_pred_list.extend(pred.argmax(1).cpu().numpy())
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if batch % print_freq == 0:
            print(f"train | loss: {loss.item():>7f}", flush=True)
    
    train_loss /= num_batches
    correct = accuracy_score(y_true=y_true_list, y_pred=y_pred_list)
    
    print(f"\n[INFO] Train Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {train_loss:>8f}\n")

    if print_report:
        print(classification_report(y_true_list, y_pred_list, digits=4))


def load_weight(model, model_path):
    print('\n==> load weight')
    weights_dict = torch.load(model_path, map_location='cpu')["model"]
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict , strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)


def update_best_model(cfg, model_state, model_name):

    result_path = cfg.result_path
    cp_path = os.path.join(result_path, model_name)

    if cfg.best_model_path is not None:
        # remove previous model weights
        os.remove(cfg.best_model_path)

    torch.save(model_state, cp_path)
    torch.save(model_state, os.path.join(result_path, "best-model.pth"))
    cfg.best_model_path = cp_path
    print(f"\n[INFO] Saved Best PyTorch Model State to {model_name}\n")


def save_cfg_and_args(result_path, cfg=None, args=None):
    """save cfg and args to file"""

    if args is not None:
        with open(os.path.join(result_path, 'args.json'), 'w') as f:
            json.dump(vars(args), f)

    if cfg is not None:
        with open(os.path.join(result_path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f)


def get_dataloaders(data_loader_type, data_name):

    if data_loader_type == 0:
        data_loaders, _ = loaders.load_data(data_name=data_name)

    return data_loaders


def get_loss_fn(args, cfg, device):

    if args.loss_type == "ce":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_type == "fl":
        print("focal gamma fator:", args.fl_gamma)
        loss_fn = partial(focal_loss, gamma=args.fl_gamma)
    elif args.loss_type == "bsl":
        sample_per_class = np.load(cfg["sample_per_class_path"])
        loss_fn = partial(balanced_softmax_loss, sample_per_class=sample_per_class)
    elif args.loss_type == "cbl":
        sample_per_class = np.load(cfg["sample_per_class_path"])
        data_name = args.data_name
        if data_name == 'cifar-10-lt-ir100':
            loss_fn = partial(CB_loss,
                          samples_per_cls=sample_per_class, 
                          no_of_classes=cfg["model"]["num_classes"], 
                          loss_type="focal", 
                          beta=0.9999,
                          gamma=2.0,
                          device=device)

    return loss_fn


def get_scheduler(lr_scheduler, optimizer, epochs):

    if lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs
        )


    return scheduler


def print_yml_cfg(cfg):
    print("")
    print("-" * 20, "yml cfg", "-" * 20)
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("-" * 18, "yml cfg end", "-" * 18, flush=True)


def draw_acc_and_loss(train_loss, test_loss,
                      train_acc, test_acc,
                      result_path, filename=None):
    history = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": test_loss,
        "val_acc": test_acc
    }

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    np.save(os.path.join(result_path, "model_acc_loss.npy" if filename is None else f"{filename}.npy"), history)

    num_epochs = len(train_loss)

    plt.plot(range(1, num_epochs + 1), train_loss, "r", label="train loss")
    plt.plot(range(1, num_epochs + 1), test_loss, "b", label="val loss")

    plt.plot(range(1, num_epochs + 1), train_acc, "g", label="train acc")
    plt.plot(range(1, num_epochs + 1), test_acc, "k", label="val acc")

    plt.title("Acc and Loss of each epoch")
    plt.xlabel("Training Epochs")
    plt.ylabel("Acc & Loss")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "model_acc_loss.jpg" if filename is None else f"{filename}.jpg"))
    plt.clf()
    plt.close()
