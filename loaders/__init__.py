
from loaders.cifar_lt_loader import load_cifar_lt_images

def load_data(data_name, data_type=None):
    print('-' * 42)
    print('LOAD DATA:', data_name)
    print('-' * 42)

    if data_type is None:
        train_loader, test_loader, train_size, test_size = None, None, None, None
        if data_name == 'cifar-10-lt-ir100':
            train_loader, train_size = load_cifar_lt_images("train", data_name)
            test_loader, test_size = load_cifar_lt_images("test", data_name)

        
        data_loaders = {'train': train_loader, 'val': test_loader}
        dataset_sizes = {'train': train_size, 'val': test_size}
        return data_loaders, dataset_sizes
    else:
        if data_name == 'cifar-10-lt-ir100':
            return load_cifar_lt_images(data_type, data_name)

        


