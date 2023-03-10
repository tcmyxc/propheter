import os
import yaml

def get_cfg(cfg_filename):
    
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    curPath = os.path.dirname(os.path.realpath(__file__))
    yamlPath = os.path.join(curPath, cfg_filename)

    with open(yamlPath, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader)
    
    return cfg