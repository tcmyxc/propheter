import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import torch
import models
import loaders
import argparse
import os
import time

from torch import optim
from configs.config_util import get_cfg
from utils.time_util import print_time, get_current_time
from sklearn.metrics import classification_report
from misc import draw_acc_and_loss
from misc import print_yml_cfg
from utils.args_util import print_args
from utils.general import update_best_model, init_seeds
from misc import get_loss_fn, get_scheduler

import warnings # ignore warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='cifar-10-lt-ir100')
parser.add_argument('--model_name', default='resnet32')
parser.add_argument('--lr', type=float, default='1e-2')
parser.add_argument('--data_loader_type', type=int, default='0')
parser.add_argument('--epochs', type=int, default='200')
parser.add_argument('--lr_scheduler', type=str, default='cosine')
parser.add_argument('--loss_type', type=str, default='bsl')
parser.add_argument('--fl_gamma', type=float, default=2.0)
parser.add_argument('--gpu_id', type=str, default='0')


def main():
    args = parser.parse_args()
    print_args(args)
    
    init_seeds()

    # get cfg
    data_name    = args.data_name
    model_name   = args.model_name
    cfg_filename = "one_stage.yml"
    cfg = get_cfg(cfg_filename)[data_name]
    print_yml_cfg(cfg)

    # result path
    result_path = os.path.join("./work_dir/baseline",
                               model_name, data_name,
                               f"lr{args.lr}", f"{args.lr_scheduler}_lr_scheduler", 
                               f"{args.loss_type}_loss",
                               get_current_time())
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    print(f"\n[INFO] result will save in:\n{result_path}\n")
    
    # add some cfg
    cfg["best_model_path"] = None
    cfg["result_path"] = result_path
    cfg["best_acc"] = 0
    cfg["g_train_loss"] = []
    cfg["g_train_acc"] = []
    cfg["g_test_loss"] = []
    cfg["g_test_acc"] = []

    lr = float(args.lr)
    momentum = cfg["optimizer"]["momentum"]
    weight_decay = float(cfg["optimizer"]["weight_decay"])
    epochs = args.epochs
    print(f"\n[INFO] total epoch: {epochs}")

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-' * 42, '\n[Info] use device:', device)

    # data loader
    if args.data_loader_type == 0:
        data_loaders, _ = loaders.load_data(data_name=data_name)

    # model
    model = models.load_model(
        model_name=model_name, 
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"]
    )
    model.to(device)

    # loss
    loss_fn = get_loss_fn(args, cfg, device)
    
    # optimizer
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # lr scheduler
    scheduler = get_scheduler(args.lr_scheduler, optimizer, epochs)

    begin_time = time.time()
    for epoch in range(epochs):
        epoch_begin_time = time.time()
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        print(f"\nEpoch {epoch+1}")
        print("[INFO] lr is:", cur_lr)
        print("-" * 42)

        train(data_loaders["train"], model, loss_fn, optimizer, device, cfg)
        test(data_loaders["val"], model, loss_fn, optimizer, epoch, device, args, cfg)
        scheduler.step()

        draw_acc_and_loss(cfg["g_train_loss"], cfg["g_test_loss"], cfg["g_train_acc"], cfg["g_test_acc"], result_path)

        print_time(time.time()-epoch_begin_time, epoch=True)
        
    print("Done!")
    print_time(time.time()-begin_time)


def train(dataloader, model, loss_fn, optimizer, device, cfg):
    train_loss, correct = 0, 0
    y_pred_list = []
    y_train_list = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(True):
            # Compute prediction error
            pred = model(X)

            loss = loss_fn(pred, y)
            train_loss += loss.item()
        
            y_pred_list.extend(pred.argmax(1).cpu().numpy())

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"train | loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)
    
    train_loss /= num_batches
    correct /= size
    cfg["g_train_loss"].append(train_loss)
    cfg["g_train_acc"].append(correct)
    print("-" * 42)
    print(classification_report(y_train_list, y_pred_list, digits=4))


def test(dataloader, model, loss_fn, optimizer, epoch, device, args, cfg):
    y_pred_list = []
    y_train_list = []
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
        
    test_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)
        with torch.set_grad_enabled(True):
            pred = model(X)
            loss = loss_fn(pred, y)

            test_loss += loss.item()

        y_pred_list.extend(pred.argmax(1).cpu().numpy())

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"val | loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)

    test_loss /= num_batches
    cfg["g_test_loss"].append(test_loss)
    correct /= size
    cfg["g_test_acc"].append(correct)
    
    if correct > cfg["best_acc"]:
        cfg["best_acc"] = correct
        print(f"\n[FEAT] Epoch {epoch+1}, update best acc:", correct)
        model_name=f"best-model-acc{correct:.4f}.pth"
        model_state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': correct,
        }
        update_best_model(cfg, model_state, model_name)
        

    print(f"\nTest Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    print(classification_report(y_train_list, y_pred_list, digits=4))



if __name__ == '__main__':
    main()