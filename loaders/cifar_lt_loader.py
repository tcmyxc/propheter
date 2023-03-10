import os

from torch.utils.data import DataLoader
from torchvision import transforms

from configs import config
from loaders.auto_aug import CIFAR10Policy, Cutout
from loaders.datasets.image_dataset import ImageDataset


def load_cifar_lt_images(data_type, dataset_name):
    print("load cifar dataset from image dir\n", flush=True)
    assert data_type in ['train', 'test']

    if dataset_name == "cifar-10-lt-ir100":
        image_dir = os.path.join("./data/cifar10_lt_ir100/images", data_type)
    
    if data_type == 'train':
        data_set = ImageDataset(
            image_dir=image_dir,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),    # add AutoAug
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )

    else:
        data_set = ImageDataset(
            image_dir=image_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )

    data_loader = DataLoader(dataset=data_set,
                             batch_size=128,
                             num_workers=4,
                             shuffle=True)

    return data_loader, len(data_set)
