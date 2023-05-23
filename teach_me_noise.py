
import shutil
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import torch
import torch.nn.functional as F
from models import resnetv3, resnetv2

import os
import os.path as osp
import time
import datetime

from torch import optim
from configs.config_util import get_cfg
from utils.time_util import print_time
from sklearn.metrics import classification_report, accuracy_score
from misc import print_yml_cfg
from utils.args_util import print_args
from utils.general import init_seeds


import warnings # ignore warnings
warnings.filterwarnings("ignore")

from misc import (evaluate, update_best_model, save_cfg_and_args,
    get_dataloaders, get_loss_fn, get_scheduler, load_weight)



def main(args):
    print_args(args)

    init_seeds(args.seed)

    # get cfg
    cfg = get_cfg(args.cfg)[args.data_name]
    print_yml_cfg(cfg)

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_name    = args.data_name
    model_name   = args.model_name
    lr           = float(args.lr)
    momentum     = cfg["optimizer"]["momentum"]
    weight_decay = float(cfg["optimizer"]["weight_decay"])
    epochs       = args.epochs

    args.result_path = os.path.join(
        args.result_path,
        f"{data_name}_{model_name}",
        f"{datetime.datetime.now().strftime('%Y%m%d/%H%M%S')}"
    )
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    print(f"\n[INFO] result path: {osp.abspath(args.result_path)}\n")

    save_cfg_and_args(args.result_path, cfg, args)
    
    # save code
    code_path = os.path.join(args.result_path, "code.py")
    cur_file_path = __file__
    shutil.copy(cur_file_path, code_path)

    # data loader
    data_loaders = get_dataloaders(args.data_loader_type, data_name)

    # teacher model
    teacher_model = resnetv3.resnet32(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"]
    )
    teacher_model.forward = teacher_model._forward_impl_v2
    teacher_model.to(device)
    print("\n[INFO] load teacher weight")
    load_weight(teacher_model, args.teacher_model_path)

    # student model
    student_model = resnetv2.resnet32(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"]
    )
    student_model.forward = student_model.forward_layer3_out
    student_model.to(device)
    if args.student_load_weight:
        print("\n[INFO] load student weight")
        load_weight(student_model, args.teacher_model_path)

    models = {
        "teacher_model": teacher_model,
        "student_model": student_model,
    }

    # loss fn
    loss_fn = get_loss_fn(args, cfg, device)
    
    # optimizer
    optimizer = optim.SGD(
        params=student_model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # lr scheduler
    scheduler = get_scheduler(args.lr_scheduler, optimizer, epochs)

    begin_time = time.time()
    best_acc = 0
    for epoch in range(epochs):
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        print(f"\nEpoch {epoch+1}")
        print(f"lr is: {cur_lr}\n")


        train_one_epoch(data_loaders["train"], models, loss_fn, optimizer, device,
                        print_report=args.print_report, print_freq=args.print_freq)
        val_acc = evaluate(data_loaders["val"], student_model, device, args)

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"\n[FEAT] best acc: {best_acc:.4f}, error rate: {(1 - best_acc):.4f}")
            model_state = {
                'epoch': epoch,
                'model': student_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acc': best_acc,
            }
            model_name=f"best-model-acc{best_acc:.4f}.pth"
            update_best_model(args, model_state, model_name)

        scheduler.step()
        
    print("Done!")
    print(f"\n[INFO] best acc: {best_acc:.4f}, error rate: {(1 - best_acc):.4f}\n")
    print_time(time.time()-begin_time)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Long-tail Classification", add_help=add_help)

    parser.add_argument('--data_name', default='cifar-10-lt-ir100')
    parser.add_argument('--model_name', default='resnet32')
    parser.add_argument('--lr', type=float, default='1e-2')
    parser.add_argument('--epochs', type=int, default='200')
    parser.add_argument('--data_loader_type', type=int, default='0')
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--loss_type', type=str, default='bsl')
    parser.add_argument('--fl_gamma', type=float, default=2.0)
    parser.add_argument('--gpu_id', type=str, default='2')
    parser.add_argument('--result_path', type=str, default='./work_dir')
    parser.add_argument('--cfg', type=str, default='one_stage.yml')
    parser.add_argument('--teacher_model_path', type=str)
    parser.add_argument('--best_model_path', action='store_const', const=None)
    parser.add_argument('--student_load_weight', action='store_true')
    parser.add_argument('--print_report', action='store_true')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)


    return parser
    

def train_one_epoch(dataloader, models, loss_fn, optimizer, device, print_report=True, print_freq=10):
    y_pred_list = []
    y_train_list = []


    train_loss = 0
    num_batches = len(dataloader)

    teacher_model = models["teacher_model"]
    student_model = models["student_model"]

    teacher_model.train()
    student_model.train()
    for batch, (X, y) in enumerate(dataloader):
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(True):
            _, f_t = teacher_model(X, y)
            pred, f_s = student_model(X)
            mse_loss = F.mse_loss(f_s, f_t)
            y_pred_list.extend(pred.argmax(1).cpu().numpy())

            loss = loss_fn(pred, y) + mse_loss
            train_loss += loss.item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % print_freq == 0:
                print(f"train | loss: {loss.item():>7f}, mse loss: {mse_loss.item():>7f}", flush=True)
    
    train_loss /= num_batches
    correct = accuracy_score(y_true=y_train_list, y_pred=y_pred_list)
    
    print(f"\nTrain Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {train_loss:>8f}")

    if print_report:
        print("-" * 42)
        print(classification_report(y_train_list, y_pred_list, digits=4))

if __name__ == "__main__":
    args = get_args_parser().parse_args()

    main(args)
    