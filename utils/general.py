"""
General utils
"""

import math
import os
import random

import numpy as np
import torch


def init_seeds(seed=0):
    """fix seed"""
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)
    
    
def update_best_model(cfg, model_state, model_name):

    result_path = cfg["result_path"]
    cp_path = os.path.join(result_path, model_name)

    if cfg["best_model_path"] is not None:
        # remove previous model weights
        os.remove(cfg["best_model_path"])

    torch.save(model_state, cp_path)
    torch.save(model_state, os.path.join(result_path, "best-model.pth"))
    cfg["best_model_path"] = cp_path
    print(f"Saved Best PyTorch Model State to {model_name} \n")
