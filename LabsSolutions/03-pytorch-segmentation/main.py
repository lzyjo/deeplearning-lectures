#!/usr/bin/env python3
# coding: utf-8
"""
    This script belongs to the lab work on semantic segmenation of the
    deep learning lectures https://github.com/jeremyfix/deeplearning-lectures
    Copyright (C) 2022 Jeremy Fix

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# Standard modules
import logging
import argparse
import pathlib
import os
import sys

# External modules
import deepcs
import deepcs.training
import deepcs.testing
import deepcs.metrics
import deepcs.display
import deepcs.fileutils
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Local modules
import data
import models


def wrap_dtype(loss):
    def wrapped_loss(inputs, targets):
        return loss(inputs, targets.long())

    return wrapped_loss


def train(args):
    """Train a neural network on the plankton classification task

    Args:
        args (dict): parameters for the training

    Examples::

        python3 main.py train --normalize --model efficientnet_b3
    """
    logging.info("Training")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Set up the train and valid transforms
    img_size = (16, 16)
    train_aug = A.Compose(
        [
            A.RandomCrop(768, 768),
            A.Resize(*img_size),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ]
    )
    valid_aug = A.Compose(
        [
            A.RandomCrop(768, 768),
            A.Resize(*img_size),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ]
    )

    def train_transforms(img, mask):
        aug = train_aug(image=np.array(img), mask=mask.numpy())
        return (aug["image"], aug["mask"])

    def valid_transforms(img, mask):
        aug = valid_aug(image=np.array(img), mask=mask.numpy())
        return (aug["image"], aug["mask"])

    train_loader, valid_loader, labels = data.get_dataloaders(
        args.datadir,
        use_cuda,
        args.batch_size,
        args.num_workers,
        args.debug,
        args.val_ratio,
        train_transforms,
        valid_transforms,
    )

    logging.info(f"Considering {len(labels)} classes : {labels}")

    # Make the model
    model = models.build_model(args.model, img_size, len(labels))
    model.to(device)

    # Make the loss
    ce_loss = wrap_dtype(nn.CrossEntropyLoss())

    # Make the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Metrics
    metrics = {"CE": ce_loss}
    val_metrics = {"CE": ce_loss}

    # Callbacks
    if args.logname is None:
        logdir = deepcs.fileutils.generate_unique_logpath(args.logdir, args.model)
    else:
        logdir = args.logdir / args.logname
    logdir = pathlib.Path(logdir)
    logging.info(f"Logging into {logdir}")

    if not logdir.exists():
        logdir.mkdir(parents=True)

    input_size = next(iter(train_loader))[0].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + " Arguments : {}".format(args)
        + "\n\n"
        + "## Summary of the model architecture\n"
        + f"{deepcs.display.torch_summarize(model, input_size)}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset}\n"
        + f"Validation : {valid_loader.dataset}"
    )

    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)

    # Callbacks
    tensorboard_writer = SummaryWriter(log_dir=logdir)
    tensorboard_writer.add_text(
        "Experiment summary", deepcs.display.htmlize(summary_text)
    )

    model_checkpoint = deepcs.training.ModelCheckpoint(
        model, os.path.join(logdir, "best_model.pt"), min_is_best=True
    )

    for e in range(args.nepochs):
        deepcs.training.train(
            model,
            train_loader,
            ce_loss,
            optimizer,
            device,
            metrics,
            num_epoch=e,
            tensorboard_writer=tensorboard_writer,
            dynamic_display=True,
        )

        test_metrics = deepcs.testing.test(model, valid_loader, device, val_metrics)
        updated = model_checkpoint.update(test_metrics["CE"])
        logging.info(
            "[%d/%d] Test:   Loss : %.3f %s"
            % (
                e,
                args.nepochs,
                test_metrics["CE"],
                "[>> BETTER <<]" if updated else "",
            )
        )

        # Metrics recording
        for m_name, m_value in test_metrics.items():
            tensorboard_writer.add_scalar(f"metrics/test_{m_name}", m_value, e)
        scheduler.step(test_metrics["CE"])


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    license = """
    main.py  Copyright (C) 2022  Jeremy Fix
    This program comes with ABSOLUTELY NO WARRANTY;
    This is free software, and you are welcome to redistribute it
    under certain conditions;
    """
    logging.info(license)
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "test"])

    parser.add_argument("--logdir", type=pathlib.Path, default="./logs")
    parser.add_argument("--datadir", type=pathlib.Path, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", choices=models.available_models, required=True)

    # Training parameters
    parser.add_argument("--logname", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--nepochs", type=int, default=50)
    parser.add_argument("--base_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    exec(f"{args.command}(args)")
