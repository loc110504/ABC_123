import os
import torch
import argparse
import numpy as np
import pandas as pd

from models import CreateModel
from data import ISICDataset, Transforms
from torch.utils.data import DataLoader
from utils import yaml_config_hook


@torch.no_grad()
def main(args):
    # ------------------
    # device & seed
    # ------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------
    # dataset & loader
    # ------------------
    transforms = Transforms(size=args.image_size)

    test_dataset = ISICDataset(
        args.data_path,
        args.csv_file_test,
        transform=transforms.test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    # ------------------
    # model (NUM CLASS = LÚC TRAIN)
    # ------------------
    model = CreateModel(
        backbone=args.backbone,
        ema=False,
        out_features=13,   # 🔥 KHÔNG LẤY TỪ DATASET
        pretrained=False
    ).to(device)

    # ------------------
    # load checkpoint
    # ------------------
    print(f"Loading checkpoint from: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()

    # ------------------
    # prediction
    # ------------------
    all_preds = []
    all_probs = []

    # ------------------
    # prediction (ONE-HOT)
    # ------------------
    all_onehot = []

    for images, _ in test_loader:
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        # model(img) -> (activations, logits)
        if isinstance(outputs, (tuple, list)):
            logits = outputs[1]
        else:
            logits = outputs

        preds = torch.argmax(logits, dim=1)   # [B]

        # one-hot
        onehot = torch.zeros(
            preds.size(0),
            13,
            device=preds.device
        )
        onehot.scatter_(1, preds.unsqueeze(1), 1)

        all_onehot.append(onehot.cpu().numpy())

    all_onehot = np.concatenate(all_onehot, axis=0)

    # ------------------
    # save one-hot csv
    # ------------------
    image_ids = test_dataset.images
    class_names = test_dataset.class_names  # đúng thứ tự train csv

    df = pd.DataFrame(all_onehot, columns=class_names)
    df.insert(0, "image", image_ids)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"\n✅ One-hot prediction saved to: {args.output_csv}")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/isic2019.yaml")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/epoch_15.pth")
    parser.add_argument("--output_csv", type=str, default="./results/test_predictions.csv")

    args, _ = parser.parse_known_args()

    # load yaml
    yaml_config = yaml_config_hook(args.config)
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    main(args)
