import os
import subprocess
import re
import time
import numpy as np
import torch
from torch.multiprocessing import set_start_method, get_start_method, get_sharing_strategy
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet34, resnet50, resnet101

print(f"Process {os.getpid()} is importing lib!")


"""Summary
Best config:
1. set_start_method('spawn')
2. persistent_workers = worker > 0

Test:
workers = 0 (persistent_workers must be false)
- fork ok
- spawn ok

workers > 0 (multi python pid in nvidia-smi)
- fork
    - persistent_workers = false : only first epoch is ok  
    - persistent_workers = true  : all epochs are ok ]
        1. cannot initialize cuda runtime in main proc
        2. cannot use two or more loaders, such as train and valid loaders
         
- spawn (if __name__ == '__main__': is needed)
    - persistent_workers = false : all epochs are ok but warns 
        1. [W CudaIPCTypes.cpp:22] Producer process has been terminated before all shared CUDA tensors released. 
        2. multi import lib 
    - persistent_workers = true  : all epochs are ok and no warnings
        1. cannot define class inside if __name__ == "__main__"
        2. cuda/runtime not released
"""


class DummyDataset(Dataset):
    def __init__(self, cuda=True):
        super(DummyDataset, self).__init__()
        self.device = 'cuda' if cuda else 'cpu'

    def __len__(self):
        return 2000

    def __getitem__(self, index):
        torch.manual_seed(index)
        # (3, 512, 512) almost equals to a (1, 96, 96, 96)
        # cuda runtime >> cuda data
        # on GTX 1080Ti, torch 1.7.0
        # cuda runtime = 667 MiB in nvidia-smi, while data = 3 MiB
        # data size
        # 1, 512, 512 * float32 = 1 MiB
        size = 512
        img = torch.ones((3, size, size), device=torch.device(self.device)) * os.getpid()
        seg = torch.ones((3, size, size), device=torch.device(self.device)) * os.getppid()
        return img, seg


def get_gpu():
    try:
        with open(os.devnull, 'w') as devnull:
            gpus = subprocess.check_output([f'nvidia-smi'],
                                           stderr=devnull).decode().rstrip('\r\n').split('\n')
        gpus = [gpu for gpu in gpus if 'python' in gpu]
        return '\n'.join(gpus)
    except Exception as e:
        return 'N/A'


if __name__ == "__main__":
    FORK = False
    VALID = True
    CACHE_RM = False
    EPOCHs = 3
    ITERS = 10
    BATCH = 8
    WORKERS = 3
    PERSISTENT_WORKERS = True and WORKERS > 0
    VALID_CUDA = True
    TRAIN_CUDA = True
    
    try:
        if FORK:
            set_start_method('fork')
        else:
            set_start_method('spawn')
        print(os.getpid(), '\n\tstart with', get_start_method(), '\n\tsharing_strategy', get_sharing_strategy())
    except RuntimeError:
        print(os.getpid(), 'error with set_start_method', 'start with', get_start_method())
        pass
    
    train_dataset = DummyDataset(TRAIN_CUDA)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH,
        num_workers=WORKERS,
        shuffle=True,
        persistent_workers=PERSISTENT_WORKERS,
    )
    print("train_dataset length", len(train_dataset))
    
    if VALID:
        valid_dataset = DummyDataset(VALID_CUDA)
        print("valid_dataset length", len(valid_dataset))
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=BATCH,
            num_workers=WORKERS,
            shuffle=True,
            persistent_workers=PERSISTENT_WORKERS,
        )

    # model = torch.nn.Conv2d(3, 1, kernel_size=1)
    model = resnet50()
    model = model.cuda()

    toc = 0
    for epoch in range(EPOCHs):
        tic = time.time()

        if toc:
            print('Epoch interval:', tic - toc)
        print('Epoch', epoch, 'at', tic)
        print(get_gpu())
        np.random.seed(epoch)

        for i, (data1, data2) in enumerate(train_loader):
            print('\t\t', 'train iter', i, 'pid=', data1.max().long().item(), 'ppid=', data2.max().long().item())
            pred = model(data1.cuda())
            pred.mean().backward()
            if CACHE_RM:
                del data1, data2, pred
                torch.cuda.empty_cache()
            if i == ITERS:
                break

        toc = time.time()
        print('train cost:', toc - tic, 'at', toc)
        print(get_gpu())

        if VALID:
            with torch.no_grad():
                for i, (data1, data2) in enumerate(valid_loader):
                    print('\t\t', 'valid iter', i, 'pid=', data1.max().long().item(), 'ppid=', data2.max().long().item())
                    pred = model(data1.cuda())
                    if CACHE_RM:
                        del data1, data2
                        torch.cuda.empty_cache()
                    if i == ITERS:
                        break
            toc = time.time()

        print('Epoch cost:', toc - tic, 'at', toc)

    print(get_gpu())