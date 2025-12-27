
import os 
import wandb
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from diffusion import diffusion_models
from misc.diffusion_utils import save_images
from dataloader import DatasetLoader
from models import UNet


def train(args):
    
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("checkpoint", args.run_name), exist_ok=True)

    dataset_loader = DatasetLoader(name= args.dataset_name, batch_size= args.batch_size, image_size= args.image_size)
    model = UNet(in_channels= 3, out_channels= 3, tdim= 256)
    diffusion = diffusion_models[args.model]()
    
    global_step = 0

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = dataset_loader.get_loader("train")

    for epoch in range(args.epochs):

        model.train()
        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{args.epochs}]",
            leave=False
        )

        for images, _ in pbar:

            t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
            loss = diffusion.diffusion_loss(model, images, t)
            loss = loss["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.use_wandb:
                wandb.log(
                    {"train/loss": loss.item()},
                    step=global_step
                )

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            global_step += 1

        model.eval()

        with torch.no_grad():
            sampled_images = diffusion.p_reverse(model, shape=images[:3].shape)

        sample_path = os.path.join("results", f"{epoch}.jpg")
        save_images(sampled_images, sample_path)

        if args.use_wandb:
            wandb.log(
                {"samples": wandb.Image(sample_path)},
                step=global_step
            )

        ckpt_path = os.path.join("checkpoint", args.run_name, "ckpt.pt")
        torch.save(model.state_dict(), ckpt_path)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="ddpm_cifar10")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="ddpm")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--model", type=str, default="ddpm")
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()
    train(args)