import torch
from models import resnetv3, resnetv2


def load_model(model_name, in_channels=3, num_classes=10):
    print('-' * 42)
    print('LOAD MODEL:', model_name)
    print('-' * 42)

    model = None
    if model_name == 'resnet32':
        model = resnetv2.resnet32(in_channels, num_classes)

    return model
