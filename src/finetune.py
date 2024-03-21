"""This script is used to fine-tune the IJEPA model on the CIFAR10 dataset using Linear Probing."""

import os
import sys
import logging

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
        self.linear_probe = torch.nn.Linear(self.target_encoder.embed_dim, num_classes)

    def forward(self, x):
        x = self.target_encoder(x)  # B x N x D
        x = x.transpose(-1, -2)  # B x D x N
        x = self.avg_pool(x).squeeze(-1)  # B x D
        x = self.linear_probe(x)
        return x


def main():
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
    batch_size = 128

    # -- optim params
    init_lr = 0.005
    weight_decay = 0.0005
    lr_scheduler_step_size = 15
    lr_scheduler_gamma = 0.1
    num_epochs = 50
    save_dir = "./output/"
    os.makedirs(save_dir, exist_ok=True)

    # -- save config
    with open(f"{save_dir}/config.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Patch Size: {patch_size}\n")
        f.write(f"Crop Size: {crop_size}\n")
        f.write(f"Num Classes: {num_classes}\n")
        f.write(f"Init LR: {init_lr}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"LR Scheduler Step Size: {lr_scheduler_step_size}\n")
        f.write(f"LR Scheduler Gamma: {lr_scheduler_gamma}\n")
        f.write(f"Num Epochs: {num_epochs}\n")
        f.write(f"Save Dir: {save_dir}\n")
        f.write(f"batch_size: {batch_size}\n")

    # -- create dataloaders
    train_loader, test_loader = create_dataloaders(
        crop_size, batch_size=batch_size, num_workers=4
    )

    # -- define model
    model = DDP(
        IJEPALinearProbe(model_name, patch_size, crop_size, num_classes).to(device)
    )
    logging.info(f"Model: \n{model}")

    # -- load weights
    checkpoint = torch.load(r_file, map_location=device)
    updated_checkpoint = {}
    for k, v in checkpoint.items():
        updated_checkpoint[k.replace("module.", "")] = v
    model.target_encoder.load_state_dict(updated_checkpoint)
    logging.info(f"Loaded weights from {r_file}")

    # -- freeze the target_encoder
    for param in model.target_encoder.parameters():
        param.requires_grad = False

    """From paper: We use a learning rate with a step-wise decay, dividing it by a factor of 10 every 15 epochs, 
    and sweep three different reference learning rates [0.01, 0.05, 0.001], and two weight decay values [0.0005, 0.0]."
    """
    # -- define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=init_lr, weight_decay=weight_decay
    )

    # -- define lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma
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
            if i % 10 == 0:
                logging.info(
                    f"Epoch {epoch}/{num_epochs} | Iter {i}/{len(train_loader)} | Train Loss: {loss.item()}"
                )

        # -- validate
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            accuracy = 100 * correct / total
            logging.info(f"Epoch {epoch} | Test Accuracy: {accuracy}")

        # -- step lr
        lr_scheduler.step()

    # -- save model
    w_file = f"{save_dir}/linear_probe_{epoch}.pth"
    torch.save(model.state_dict(), w_file)
    logging.info(f"Saved weights to {w_file}")


if __name__ == "__main__":
    main()
