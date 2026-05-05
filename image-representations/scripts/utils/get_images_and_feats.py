# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn as nn
from tqdm import tqdm


def get_training_images_and_feats(
    device: str,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
):
    """Collects images and features from the model.
    """

    data = []
    labels = []
    feats = []

    # set module to eval model and collect all feature representations
    model.eval()
    with torch.no_grad():
        for x in tqdm(dataloader, desc="Collecting features"):
            x = x[0].to(device, non_blocking=True) # only take first view

            data.append(x.cpu())
            feat = model(x)["z"]
            feats.append(feat.cpu())
    model.train()

    data = torch.cat(data, dim=0).numpy()
    feats = torch.cat(feats, dim=0).numpy()

    return data, feats

def get_images_and_feats(
    device: str,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
):
    """Collects images and features from the model.
    """

    data = []
    labels = []
    feats = []

    # set module to eval model and collect all feature representations
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Collecting features"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            data.append(x.cpu())
            labels.append(y.cpu())
            feat = model(x)["z"]
            feats.append(feat.cpu())
    model.train()

    data = torch.cat(data, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    feats = torch.cat(feats, dim=0).numpy()

    return data, labels, feats

def get_images_and_feats_and_pure_labels(
    device: str,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
):
    """Collects images and features from the model.
    """

    data = []
    labels = []
    feats = []

    # set module to eval model and collect all feature representations
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Collecting features"):
            x = x.to(device, non_blocking=True)

            data.append(x.cpu())
            labels.append(y)
            feat = model(x)["z"]
            feats.append(feat.cpu())
    model.train()

    data = torch.cat(data, dim=0).numpy()
    feats = torch.cat(feats, dim=0).numpy()

    return data, labels, feats

def get_images_and_feats_and_pure_labels_train(
    device: str,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
):
    """Collects images and features from the model.
    """

    data = []
    labels = []
    feats = []

    # set module to eval model and collect all feature representations
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Collecting features"):
            img1, img2 = x
            x = img1.to(device, non_blocking=True) # only take first view

            data.append(x.cpu())
            labels.append(y)
            feat = model(x)["z"]
            feats.append(feat.cpu())
    model.train()

    data = torch.cat(data, dim=0).numpy()
    feats = torch.cat(feats, dim=0).numpy()

    return data, labels, feats


def get_images_and_feats_and_vars(
    device: str,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
):
    """Collects images and features from the model.
    """

    data = []
    labels = []
    feats = []
    vars = []

    # set module to eval model and collect all feature representations
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Collecting features"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            data.append(x.cpu())
            labels.append(y.cpu())
            out = model(x)
            feat = out["z"]
            var = out["vars"]
            feats.append(feat.cpu())
            vars.append(var.cpu())
    model.train()

    data = torch.cat(data, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    feats = torch.cat(feats, dim=0).numpy()
    vars = torch.cat(vars, dim=0).numpy()

    return data, labels, feats, vars