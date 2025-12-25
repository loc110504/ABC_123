# train the encoder (SINGLE GPU VERSION)
import os
import time
import torch
import wandb
import torch.nn as nn

from utils.loss import ProbabilityLoss, BatchLoss, ChannelLoss
from utils import epochVal, classwise_evaluation


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def trainEncoder(model, ema_model, dataloader, optimizer, logger, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== losses =====
    probability_loss_func = ProbabilityLoss()
    batch_sim_loss_func = BatchLoss(args.batch_size, world_size=1)
    channel_sim_loss_func = ChannelLoss(args.batch_size, world_size=1)
    classification_loss_func = nn.CrossEntropyLoss()

    start = time.time()
    cur_iters = 0
    model.train()

    train_loader, val_loader, test_loader = dataloader
    cur_lr = args.lr

    for epoch in range(args.epochs):
        for i, ((img, ema_img), label) in enumerate(train_loader):
            img = img.to(device)
            ema_img = ema_img.to(device)
            label = label.to(device)

            # ===== forward =====
            activations, outputs = model(img)
            with torch.no_grad():
                ema_activations, ema_output = ema_model(ema_img)

            # ===== losses =====
            classification_loss = classification_loss_func(outputs, label)

            probability_loss = (
                torch.sum(probability_loss_func(outputs, ema_output))
                / args.batch_size
            )

            batch_sim_loss = torch.sum(
                batch_sim_loss_func(activations, ema_activations)
            )

            channel_sim_loss = torch.sum(
                channel_sim_loss_func(activations, ema_activations)
            )

            loss = classification_loss * args.classification_loss_weight

            if epoch > 20:
                loss = (
                    loss
                    + probability_loss * args.probability_loss_weight
                    + batch_sim_loss * args.batch_loss_weight
                    + channel_sim_loss * args.channel_loss_weight
                )

            # ===== backward =====
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===== EMA update =====
            update_ema_variables(
                model, ema_model, args.ema_decay, cur_iters
            )

            cur_iters += 1

            # ===== logging =====
            if cur_iters % 10 == 0:
                cur_lr = optimizer.param_groups[0]["lr"]

                val_metrics = epochVal(model, val_loader)
                test_metrics = epochVal(model, test_loader)

                if logger is not None:
                    logger.log({
                        "train/total_loss": loss.item(),
                        "train/classification_loss": classification_loss.item(),
                        "train/probability_loss": probability_loss.item(),
                        "train/batch_similarity_loss": batch_sim_loss.item(),
                        "train/channel_similarity_loss": channel_sim_loss.item(),
                        "lr": cur_lr,
                        "val/Accuracy": val_metrics[0],
                        "val/F1": val_metrics[1],
                        "val/AUC": val_metrics[2],
                        "val/BAC": val_metrics[3],
                        "test/Accuracy": test_metrics[0],
                        "test/F1": test_metrics[1],
                        "test/AUC": test_metrics[2],
                        "test/BAC": test_metrics[3],
                    })

                print(
                    f"\rEpoch [{epoch+1}/{args.epochs}] "
                    f"Iter [{i+1}/{len(train_loader)}] "
                    f"lr={cur_lr:.6f} "
                    f"Loss={loss.item():.4f}",
                    end="",
                    flush=True,
                )

        # ===== save checkpoint =====
        save_path = os.path.join(
            args.checkpoints, f"epoch_{epoch+1}.pth"
        )
        torch.save(model.state_dict(), save_path)

    # ===== final per-class evaluation =====
    df = classwise_evaluation(model, test_loader)
    if logger is not None:
        logger.log({"Perclass performance": wandb.Table(dataframe=df)})
    else:
        print(df)
