import argparse
import sys
from configparser import ConfigParser

import numpy as np
import torch
from tabulate import tabulate
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dataset import denorm_batch, load_dataset
from loss_utils import sample_from_dmol
from utils import (
    LossDict,
    ModelConfig,
    TrainConfig,
    create_dq,
    get_additional_args,
    load_dq,
    mkdir,
    pjoin,
)

_LOSS_NAMES = ("total", "recon", "codebook")


def test_dq(model, testloader, device, cutoff=float("inf")):
    model.eval()
    loss = LossDict(_LOSS_NAMES, n_samples=len(testloader) + 1)
    with torch.no_grad():
        for i, x in enumerate(testloader):
            img, labels = x
            out, _, loss_dict = model_step(model, img, device)
            loss.update(**loss_dict)
            if i > cutoff:
                break
    model.train()
    return loss


def model_step(model, img, device):
    img = img.to(device)
    out, _, (loss, recon_loss, latent_loss) = model(img)
    loss_dict = dict(
        zip(_LOSS_NAMES, [loss.item(), recon_loss.item(), latent_loss.item()])
    )
    return out, loss, loss_dict


def generate_reconstructions(img, out, sample_size, loss_name):
    original = img.cpu()[:sample_size]
    original = denorm_batch(original)
    original = torch.true_divide(original, 255)
    generated = out[:sample_size]
    if loss_name == "ce":
        B, _, W, H = generated.shape
        generated = generated.view(B, 256, -1, W, H)
        generated = generated.argmax(1)
        generated = torch.true_divide(generated, 255)
    elif loss_name == "mse":
        generated = denorm_batch(generated)
        generated = torch.true_divide(generated, 255)
    elif loss_name == "mix":
        generated = sample_from_dmol(generated)
        generated = (generated + 1) / 2
    else:
        raise NotImplementedError("Unimplemneted loss function for sampling")
    generated = generated.cpu()
    out = torch.cat([original, generated], 0)
    return make_grid(out, normalize=False, nrow=sample_size)


def train_dq(model, optimizer, scaler, config, device, trainloader, testloader):
    train_config, model_config = config
    stats = train_config.stats
    cur_epoch = stats["cur_epoch"]
    iters = stats["iters"]
    best_loss = stats["best_loss"]
    train_len = len(trainloader)
    eval_itr = int(train_config.eval_itr * train_len)
    epochs = train_config.epochs
    save_dir = train_config.save_dir

    # only creates directories if they don't exist. Returns path to directory
    sample_dir = mkdir(save_dir, "sample")
    ckpt_dir = mkdir(save_dir, "checkpoint")
    log_dir = mkdir(save_dir, "log")

    test_loss = LossDict(_LOSS_NAMES)
    train_loss = LossDict(_LOSS_NAMES, n_samples=eval_itr)

    for epoch in range(cur_epoch, epochs):
        model.train()

        tqdm_trainloader = tqdm(trainloader)

        for x in tqdm_trainloader:

            img, labels = x
            model.zero_grad()
            with torch.cuda.amp.autocast(enabled=train_config.use_amp):
                out, loss, loss_dict = model_step(model, img, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss.update(**loss_dict)

            if (iters % eval_itr == 0) and iters >= eval_itr:
                cur_loss_average = train_loss.get(0)
                batch_size = img.shape[0]
                sample_size = batch_size if batch_size < 32 else 32
                img_out = generate_reconstructions(
                    img, out, sample_size, model.loss_name
                )
                image_path = pjoin(sample_dir, f"{str(iters).zfill(8)}.png")
                save_image(img_out, image_path)
                stats["iters"] = iters + 1
                stats["cur_epoch"] = epoch
                stats["train_loss"] = cur_loss_average
                save_dict = {
                    "model": model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict(),
                    "scaler": scaler.state_dict(),
                    "stats": stats,
                    "model_config": model_config,
                    "optimizer": optimizer.state_dict(),
                }

                save_file = pjoin(ckpt_dir, f"chkpt_{str(iters).zfill(8)}.pt")

                torch.save(save_dict, save_file)
                if testloader:
                    cutoff = (
                        int(train_config.test_cutoff)
                        if hasattr(train_config, "test_cutoff")
                        else len(trainloader) * 0.01
                    )
                    test_loss = test_dq(model, testloader, device, cutoff)

                if test_loss.get(0) > 0 and test_loss.get(0) < best_loss:
                    best_loss = test_loss.get(0)
                    save_dict["stats"]["best_loss"] = best_loss
                    save_file = pjoin(save_dir, f"best_model.pt")
                    torch.save(save_dict, save_file)

                writer = SummaryWriter(log_dir=log_dir)
                writer.add_scalars("Train Losses", train_loss.get_all(), iters)
                writer.add_scalars("Test Losses", test_loss.get_all(), iters)
                writer.flush()
                writer.close()

            desc = f"epoch: {epoch + 1}; iter: {iters}; Train -- {train_loss} Test -- {test_loss}"

            tqdm_trainloader.set_description(desc)
            iters += 1
        cur_epoch = epoch


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--load", dest="load", default=False, action="store_true")
    parser.add_argument(
        "--device",
        choices=(["cpu", "cuda"] + ["cuda:%s" % i for i in range(8)]),
        default="cuda",
    )

    flags, config_args = parser.parse_known_args()
    config = ConfigParser()
    config.read(flags.config)
    train_config = TrainConfig()
    model_config = ModelConfig()
    train_config.load(**dict(config["train"]))
    model_config.load(**dict(config["model"]))
    additional_args = get_additional_args(config_args)
    train_config.load(**additional_args)

    device = flags.device
    trainloader, testloader = load_dataset(
        train_config.dataset,
        train_config.image_size,
        train_config.batch_size,
        drop_last=False,
    )
    train_config.eval_itr = (
        train_config.eval_itr if train_config.eval_itr > 0 else len(trainloader)
    )

    if flags.load:
        # load latest checkpoint from checkpoint folder
        model, optimizer, scaler, model_config, train_config.stats = load_dq(
            device=device,
            save_dir=train_config.save_dir,
            save_path=train_config.save_path,
            use_amp=train_config.use_amp,
        )

    else:
        model = create_dq(model_config)
        model.to(device)
        scaler = torch.cuda.amp.GradScaler(enabled=train_config.use_amp)
        optimizer = optim.Adam(model.parameters(), lr=train_config.lr)

    print("Train Configuration: %s" % train_config)
    print("Model Configuration: %s" % model_config)

    img, labels = next(iter(trainloader))
    img = img[:1]
    model.eval()
    _, codes, _ = model(img.to(device))
    quantizers = list(reversed(model.quantizers))
    table = []
    for n_hier, code in enumerate(codes):
        dims = np.array(code.shape[1:])
        K = quantizers[n_hier][0].n_embed
        table.append([n_hier, dims, K])
    print(tabulate(table, headers=["Hierarchy", "Code Dim (M,W,H)", "K"]))
    model.train()
    model = model.to(device)
    train_dq(
        model,
        optimizer,
        scaler,
        (train_config, model_config),
        device,
        trainloader,
        testloader,
    )
    sys.exit(1)
