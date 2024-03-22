"""This script is used to fine-tune the IJEPA model on the CIFAR10 dataset using Linear Probing."""

import os
import sys
import logging
from tqdm import tqdm

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s"
)
logger = logging.getLogger()

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models import vision_transformer as vit
from src.utils.distributed import init_distributed


def create_dataloaders(crop_size, batch_size, num_workers):
    # load Cifar100 train and val datasets

    # Ref https://github.com/facebookresearch/ijepa/blob/main/src/transforms.py
    # No horizontal_flip, color_distortion, gaussian_blur
    # cifar100_mean = (0.5071, 0.4867, 0.4408)
    # cifar100_std = (0.2675, 0.2565, 0.2761)
    cifar100_mean, cifar100_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    # According to paper, resize image to 224x224 to match with pretraining size of IJEPA
    crop_scale = (0.3, 1.0)
    transform_train = torchvision.transforms.Compose(
        [
            transforms.RandomResizedCrop(crop_size, scale=crop_scale),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
        ]
    )
    train_dataset = datasets.CIFAR100(
        root="data", train=True, transform=transform_train, download=True
    )
    test_dataset = datasets.CIFAR100(
        root="data", train=False, transform=transform_test, download=True
    )

    # define dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, test_loader


class IJEPALinearProbe(torch.nn.Module):
    def __init__(self, model_name, patch_size, crop_size, num_classes):
        super(IJEPALinearProbe, self).__init__()
        self.target_encoder = vit.__dict__[model_name](
            img_size=[crop_size], patch_size=patch_size
        )  # B x N x D
        # average pool the N patches to get a single feature vector of shape B x D
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.bn = torch.nn.BatchNorm1d(4 * self.target_encoder.embed_dim)
        self.linear_probe = torch.nn.Linear(
            4 * self.target_encoder.embed_dim, num_classes
        )

    def forward(self, x):
        _, x = self.target_encoder(x)  # B x N x 4D
        x = x.transpose(-1, -2)  # B x 4D x N
        x = self.avg_pool(x).squeeze(-1)  # B x 4D
        x = self.linear_probe(self.bn(x))
        return x


def main():

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- define config
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # -- meta
    r_file = "./weights/target_encoder.pth"
    model_name = "vit_huge"
    patch_size = 14
    crop_size = 224
    num_classes = 100
    batch_size = 256

    # -- optim params
    init_lr = 0.01
    weight_decay = 0.0005
    num_epochs = 5
    save_dir = "./output/cifar100_linear_probe_4_bn"
    os.makedirs(save_dir, exist_ok=True)

    # -- save config
    with open(f"{save_dir}/config.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Patch Size: {patch_size}\n")
        f.write(f"Crop Size: {crop_size}\n")
        f.write(f"Num Classes: {num_classes}\n")
        f.write(f"Init LR: {init_lr}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"Num Epochs: {num_epochs}\n")
        f.write(f"Save Dir: {save_dir}\n")
        f.write(f"batch_size: {batch_size}\n")

    # -- create dataloaders
    train_loader, test_loader = create_dataloaders(
        crop_size, batch_size=batch_size, num_workers=4
    )

    # -- define model
    model = DDP(
        IJEPALinearProbe(model_name, patch_size, crop_size, num_classes).to(device),
        find_unused_parameters=True,
    )
    logging.info(f"Model: \n{model}")

    # -- load weights
    checkpoint = torch.load(r_file, map_location=device)
    updated_checkpoint = {}
    for k, v in checkpoint.items():
        updated_checkpoint[k.replace("module.", "")] = v
    model.module.target_encoder.load_state_dict(updated_checkpoint)
    logging.info(f"Loaded weights from {r_file}")

    # -- freeze the target_encoder
    for param in model.module.target_encoder.parameters():
        param.requires_grad = False

    """From paper: We use a learning rate with a step-wise decay, dividing it by a factor of 10 every 15 epochs, 
    and sweep three different reference learning rates [0.01, 0.05, 0.001], and two weight decay values [0.0005, 0.0]."
    """
    # -- define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=init_lr, weight_decay=weight_decay
    )

    # -- define lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=len(train_loader), T_mult=1, eta_min=0.0001
    )

    for epoch in range(num_epochs):
        # -- train
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch + i / len(train_loader))
            if i % 10 == 0 or i == len(train_loader) - 1:
                logging.info(
                    f"Epoch {epoch}/{num_epochs} | Iter {i}/{len(train_loader)} | Train Loss: {loss.item()}"
                )

        # --validate
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            accuracy = 100 * correct / total
            logging.info(f"Epoch {epoch} | Test Accuracy: {accuracy}")

        # -- save model, optimizer, lr_scheduler for rank 0
        if rank == 0:
            checkpoint = {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "epoch": epoch,
                "accuracy": accuracy,
            }
            w_file = f"{save_dir}/checkpoint_{epoch}.pth"
            torch.save(checkpoint, w_file)
            logging.info(f"Saved checkpoint to {w_file}")


if __name__ == "__main__":
    main()
