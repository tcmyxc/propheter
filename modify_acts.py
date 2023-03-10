
import shutil
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import torch
from models.resnetv3 import resnet32
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

from misc import evaluate, update_best_model, save_cfg_and_args, \
    get_dataloaders, get_loss_fn, get_scheduler



def main(args):
    print_args(args)

    init_seeds()

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
    
    code_path = os.path.join(args.result_path, "code.py")
    cur_file_path = __file__
    shutil.copy(cur_file_path, code_path)

    # data loader
    data_loaders = get_dataloaders(args.data_loader_type, data_name)

    # model
    model = resnet32(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"]
    )
    model.to(device)

    # loss fn
    loss_fn = get_loss_fn(args, cfg, device)
    
    # optimizer
    optimizer = optim.SGD(
        params=model.parameters(),
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

        if epoch % 7 == 0:
            print("[INFO] add noise\n")
            add_noise = True
        else:
            add_noise = False

        train_one_epoch(data_loaders["train"], model, loss_fn, optimizer, device,
                        print_report=args.print_report, print_freq=args.print_freq, 
                        add_noise=add_noise)
        val_acc = evaluate(data_loaders["val"], model, device, args)

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"\n[FEAT] best acc: {best_acc:.4f}, error rate: {(1 - best_acc):.4f}")
            model_state = {
                'epoch': epoch,
                'model': model.state_dict(),
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
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--result_path', type=str, default='./work_dir')
    parser.add_argument('--threshold', type=float, default='0.5')
    parser.add_argument('--cfg', type=str, default='one_stage.yml')
    parser.add_argument('--best_model_path', action='store_const', const=None)
    parser.add_argument('--print_report', action='store_true')
    parser.add_argument('--print_freq', type=int, default=10)

    return parser
    

def train_one_epoch(dataloader, model, loss_fn, optimizer, device, print_report=True, print_freq=10, add_noise=True):
    y_pred_list = []
    y_train_list = []

    train_loss = 0
    num_batches = len(dataloader)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(True):
            if add_noise:
                pred = model(X, y)
            else:
                pred = model(X)
            y_pred_list.extend(pred.argmax(1).cpu().numpy())

            loss = loss_fn(pred, y)
            train_loss += loss.item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % print_freq == 0:
                print(f"train | loss: {loss.item():>7f}", flush=True)
    
    train_loss /= num_batches
    correct = accuracy_score(y_true=y_train_list, y_pred=y_pred_list)
    
    print(f"\nTrain Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {train_loss:>8f}")

    if print_report:
        print("-" * 42)
        print(classification_report(y_train_list, y_pred_list, digits=4))

if __name__ == "__main__":
    args = get_args_parser().parse_args()

    main(args)
    