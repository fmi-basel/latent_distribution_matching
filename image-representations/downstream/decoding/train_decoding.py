import os
import json
from omegaconf import OmegaConf
import argparse
from solo.methods import METHODS
from solo.data.classification_dataloader import prepare_data
from scripts.utils.get_images_and_feats import get_images_and_feats, get_training_images_and_feats
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import torch.nn as nn
import torch

class CifarDecoder(nn.Module):
    def __init__(self, latent_dim, initial_feature_map_size=(128, 4, 4), output_channels=3):
        super(CifarDecoder, self).__init__()

        self.initial_feature_map_size = initial_feature_map_size
        a, b, c = initial_feature_map_size
        self.latent_dim = latent_dim
        self.output_channels = output_channels

        # Map latent vector to a small spatial feature map (e.g., 128 x 4 x 4)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, a * b * c),
            # nn.ReLU(),
            # nn.Linear(a * b * c, a * b * c),
            # nn.ReLU(),
        )
        self.conv_layers = nn.Sequential(
            # Reshape to (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Upsample to (32, 8, 8)
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Upsample to (16, 16, 16)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Upsample to (3, 32, 32)
            nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output pixel values in [0, 1]
        )

    def forward(self, z):
        a, b, c = self.initial_feature_map_size
        x = self.fc(z)
        x = x.view(-1, a, b, c)  # Reshape to feature map
        x = self.conv_layers(x)
        return x

def get_training_data(model_name, run_name, dataset, 
                      train_data_path, val_data_path, data_format, 
                      batch_size, num_workers, shuffle=True):
    
    folder_name = "../../trained_models/" + model_name + "/"
    # get name of the most recent model by folder creation time
    names = sorted(os.listdir(folder_name), key=lambda x: os.path.getctime(os.path.join(folder_name, x)))
    for n in names:
        run = folder_name + n
        args_file = os.path.join(run, "args.json")
        args = json.load(open(args_file, "r"))
        if args["name"] == run_name:
            pretrained_checkpoint_dir = run
            name = n
    cfg = OmegaConf.create(args)

    # build paths
    ckpt_dir = Path(pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"
    ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)
    cfg = OmegaConf.create(method_args)

    # build the model
    model = (
        METHODS[method_args["method"]]
        .load_from_checkpoint(ckpt_path, strict=False, cfg=cfg)
    )
    # prepare data
    train_loader, val_loader = prepare_data(
        dataset,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        data_format=data_format,
        batch_size=batch_size,
        num_workers=num_workers,
        auto_augment=False,
        shuffle=shuffle,
        no_train_transform=True,
    )

    # move model to the gpu
    device = "cuda:0"
    model = model.to(device)

    # get images and features
    train_data, train_z = get_training_images_and_feats(device, model, train_loader)
    train_z = train_z.reshape((train_z.shape[0], -1))
    val_data, _, val_z = get_images_and_feats(device, model, val_loader)
    val_z = val_z.reshape((val_z.shape[0], -1))

    # rescale images to [0, 1]
    train_data = (train_data + 2.0) / 4.0
    train_data = train_data.clip(0, 1)
    val_data = (val_data + 2.0) / 4.0
    val_data = val_data.clip(0, 1)

    return model, train_data, train_z, val_data, val_z

def train_decoder(train_data, train_z, val_data, val_z, batch_size=256, num_epochs=2, lr=1e-3):
    # define a simple decoder
    latent_dim = train_z.shape[1]
    decoder = CifarDecoder(latent_dim=latent_dim).to("cuda:0")

    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    # create a dataloader for training data
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_z, dtype=torch.float32), 
                                                   torch.tensor(train_data, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # create a dataloader for validation data
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_z, dtype=torch.float32),
                                                torch.tensor(val_data, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # training loop
    for epoch in range(num_epochs):
        decoder.train()
        running_train_loss = 0.0
        for i, (z_batch, img_batch) in enumerate(train_loader):
            z_batch = z_batch.to("cuda:0")
            img_batch = img_batch.to("cuda:0")

            optimizer.zero_grad()
            outputs = decoder(z_batch)
            loss = criterion(outputs, img_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        epoch_train_loss = running_train_loss / len(train_loader)
        
        with torch.no_grad():
            decoder.eval()
            val_loss = 0.0
            for i, (z_batch, img_batch) in enumerate(val_loader):
                z_batch = z_batch.to("cuda:0")
                img_batch = img_batch.to("cuda:0")
                outputs = decoder(z_batch)
                loss = criterion(outputs, img_batch)
                val_loss += loss.item()
            val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    return decoder

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for data loader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loader")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train the decoder")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for optimizer")

    args = parser.parse_args()
    model_name = args.model_name
    run_name = args.run_name
    batch_size = args.batch_size
    num_workers = args.num_workers
    dataset = args.dataset
    val_data_path = "../../datasets/imagenet100/val"
    train_data_path = "../../datasets/"
    data_format = "image_folder"

    model, train_data, train_z, val_data, val_z = \
        get_training_data(model_name, 
                          run_name, dataset, 
                          train_data_path, 
                          val_data_path, 
                          data_format, 
                          batch_size, 
                          num_workers)
    
    decoder = train_decoder(train_data, train_z, 
                            val_data, val_z,
                            batch_size=batch_size, 
                            num_epochs=args.num_epochs, 
                            lr=args.lr)
    
    # save decoder model
    save_path = f"./decoders/{model_name}_{run_name}_decoder.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(decoder.state_dict(), save_path)
    print(f"Decoder model saved to {save_path}")
