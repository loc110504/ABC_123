import os
import torch
import argparse
import torch.distributed as dist
from models import CreateModel
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from data import ISICDataset, Transforms
from torch.utils.data import DataLoader
from utils import trainEncoder, yaml_config_hook
from utils.sync_batchnorm import convert_model
from prepare_datasets import construct_ISIC2019LT


def main(args):
    # ------------------
    # device & seed
    # ------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------
    # dataset
    # ------------------
    transforms = Transforms(size=args.image_size)

    train_dataset = ISICDataset(
        args.data_path,
        args.csv_file_train,
        transform=transforms
    )

    val_dataset = ISICDataset(
        args.data_path,
        args.csv_file_val,
        transform=transforms.test_transform
    )

    test_dataset = ISICDataset(
        args.data_path,
        args.csv_file_test,
        transform=transforms.test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    loaders = (train_loader, val_loader, test_loader)

    # ------------------
    # model
    # ------------------
    num_class = train_dataset.n_class

    model = CreateModel(
        backbone=args.backbone,
        ema=False,
        out_features=num_class,
        pretrained=args.pretrained
    ).to(device)

    ema_model = CreateModel(
        backbone=args.backbone,
        ema=True,
        out_features=num_class,
        pretrained=args.pretrained
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9
    )

    # ------------------
    # DataParallel (optional)
    # ------------------
    if args.dataparallel and torch.cuda.device_count() > 1:
        model = DataParallel(model)
        ema_model = DataParallel(ema_model)

    # ------------------
    # train
    # ------------------
    trainEncoder(
        model,
        ema_model,
        loaders,
        optimizer,
        logger=None,
        args=args
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/isic2019.yaml')
    args, _ = parser.parse_known_args()

    yaml_config = yaml_config_hook(args.config)
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    os.makedirs(args.checkpoints, exist_ok=True)

    main(args)
