from tqdm import tqdm
from IPython.display import clear_output

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler


def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
    train_pnls: list[float],
    val_pnls: list[float],
):
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label="train")
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label="val")
    axs[0].set_ylabel("loss")

    axs[1].plot(range(1, len(train_pnls) + 1), train_pnls, label="train")
    axs[1].plot(range(1, len(val_pnls) + 1), val_pnls, label="val")
    axs[1].set_ylabel("PnL, RUB")

    for ax in axs:
        ax.set_xlabel("epoch")
        ax.legend()

    plt.show()


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    loader: DataLoader,
    tqdm_desc: str = "Model",
):
    device = next(model.parameters()).device

    if device == torch.device("cuda"):
        has_cuda = True
    else:
        has_cuda = False

    if tqdm_desc is None:
        iterator = loader
    else:
        iterator = tqdm(loader, desc=tqdm_desc)

    if has_cuda:
        scaler = GradScaler()

    train_loss = 0.0
    model_diff = 0.0
    model.train()
    weight_path = []
    diffs_path = []
    for features, target_pnl in iterator:
        optimizer.zero_grad()

        features = features.to(device)
        target_pnl = target_pnl.to(device)

        if has_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                weights, model_pnl = model.get_pnl(features)
                loss = criterion(target_pnl, model_pnl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
        else:
            weights, model_pnl = model.get_pnl(features)
            loss = criterion(target_pnl, model_pnl)

            loss.backward()
            optimizer.step()

        diff = target_pnl - model_pnl

        train_loss += loss.item()
        model_diff += diff.mean().item()

        diffs_path.append(diff.detach().cpu().numpy())
        weight_path.append(weights.detach().cpu().numpy())

    train_loss /= len(loader.dataset)
    model_diff /= len(loader.dataset)

    return train_loss, weight_path, model_diff, diffs_path


@torch.no_grad()
def validation_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    tqdm_desc: [str, None] = None,
):
    device = next(model.parameters()).device

    if tqdm_desc is None:
        iterator = loader
    else:
        iterator = tqdm(loader, desc=tqdm_desc)

    val_loss = 0.0
    model_diff = 0.0
    model.eval()
    diffs_path = []
    weight_path = []
    for features, target_pnl in iterator:
        features = features.to(device)
        target_pnl = target_pnl.to(device)

        weights, model_pnl = model.get_pnl(features)

        loss = criterion(target_pnl, model_pnl)
        diff = target_pnl - model_pnl

        val_loss += loss.item()
        model_diff += diff.mean().item()

        diffs_path.append(diff.detach().cpu().numpy())
        weight_path.append(weights.detach().cpu().numpy())

    val_loss /= len(loader.dataset)
    model_diff /= len(loader.dataset)

    return val_loss, weight_path, model_diff, diffs_path
