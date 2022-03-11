import glob
import os
import pathlib
from collections import OrderedDict

import numpy as np
import torch


class MovingAverage:
    """
    Moving average
    """

    def __init__(self, limit=3000):

        self.limit = limit
        self.vals = []
        self.avg = 0

    def update(self, x):
        self.vals.append(x)
        if len(self.vals) > self.limit:
            self.vals.pop(0)

        self.avg = np.mean(self.vals)

    def get(self):
        return self.avg


def pjoin(path, *paths):
    return os.path.join(path, *paths)


def save_model(model, optimizer, scaler, config, stats, save_file):
    save_dict = {
        "model": model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict(),
        "scaler": scaler,
        "stats": stats,
        "model_config": config,
        "optimizer": optimizer.state_dict(),
    }

    torch.save(save_dict, save_file)


def mkdir(path, *paths):
    pathlib.Path(pjoin(path, *paths)).mkdir(parents=True, exist_ok=True)
    return pjoin(path, *paths)


def get_additional_args(config_args):
    keys = [c[2:] for c in (config_args[::2])]
    values = config_args[1::2]
    additional_args = dict(zip(keys, values))
    return additional_args


def get_chkpt_len(save_dir):

    chkpt_dir = pjoin(save_dir, "checkpoint", "*")
    sorted_chkpts = sorted(glob.glob(chkpt_dir))
    return len(sorted_chkpts)


def get_chkpt_path(save_dir, chkpt_index=-1):
    chkpt_dir = pjoin(save_dir, "checkpoint", "*")
    sorted_chkpts = sorted(glob.glob(chkpt_dir))
    if len(sorted_chkpts) > 0:
        return sorted_chkpts[chkpt_index]

    raise Exception("Checkpoint not found on %s" % save_dir)


def load_chkpt_dict(save_dir=None, save_path=None):
    import torch

    if save_dir and len(save_dir) > 0:
        is_valid_chkpt = False
        chkpt_len = get_chkpt_len(save_dir)
        i = 1
        while not is_valid_chkpt:
            chkpt_path = get_chkpt_path(save_dir, chkpt_index=-i)
            print("Loading: %s" % chkpt_path)
            ckpt_dict = torch.load(chkpt_path, map_location="cpu")
            stats = ckpt_dict["stats"]

            is_valid_chkpt = not np.isnan(stats["train_loss"])
            if not is_valid_chkpt:
                print("Nan Loss: %s" % chkpt_path)
            if i > chkpt_len:
                raise Exception("No Valid Checkpoint found. Loss was nan.")

            i += 1
        load_path = chkpt_path

    elif save_path and len(save_path) > 0:
        load_path = str(save_path)

        print("Loading: %s" % load_path)
        ckpt_dict = torch.load(load_path, map_location="cpu")
        stats = ckpt_dict["stats"]
        if np.isnan(stats["train_loss"]):
            raise Exception("Invalid checkpoint")
    else:
        raise Exception("No chkpt passed")
    return ckpt_dict, load_path


class LossDict:
    def __init__(self, loss_names, n_samples=1000):

        self.loss_names = loss_names
        self.data = OrderedDict()
        for _name in loss_names:
            self.data[_name.lower()] = MovingAverage(limit=n_samples)

    def update(self, *args, **kwargs):
        for k, v in kwargs.items():
            self.data[k.lower()].update(v)

    def get(self, index):
        return list(self.data.values())[index].get()

    def get_all(self):
        return {k: v.get() for k, v in self.data.items()}

    def __repr__(self):
        return " ".join([f"{k}: {v.get():.2e}; " for k, v in self.data.items()])


class BaseConfig:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def load(self, *args, **config):

        for k, v in config.items():
            attr_type = type(getattr(self, k, ""))
            if v == None:

                setattr(self, k, None)
            else:
                if attr_type == bool:

                    setattr(self, k, bool(eval(v)))
                else:
                    setattr(self, k, attr_type(v))

    def __repr__(self):
        return str(self.__dict__)


class ModelConfig(BaseConfig):
    in_channel = 3
    channel = 256
    n_res_block = 3
    n_res_channel = 256
    n_coder_blocks = 2
    embed_dim = 64
    n_codebooks = 5
    stride = 2
    decay = 0.99
    loss_name = "mse"
    vq_type = "dq"
    beta = 0.25
    n_hier = "[512]"
    n_logistic_mix = 10

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestConfig(BaseConfig):
    model_path = ""
    dataset = "cifar10"
    use_amp = False
    image_size = 32
    batch_size = 128
    stats = {"cur_epoch": 0, "iters": 0, "best_loss": float("inf")}

    def __init__(self, *args, **kwargs):
        super(TestConfig, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


class TrainConfig(BaseConfig):

    dataset = "cifar10"
    save_dir = ""
    save_path = ""
    device = "cuda"
    epochs = 200
    eval_itr = -1.0
    use_amp = True
    image_size = 32
    batch_size = 128
    lr = 2e-4
    stats = {"cur_epoch": 0, "iters": 0, "best_loss": float("inf")}

    def __init__(self, *args, **kwargs):
        super(TrainConfig, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


def create_dq(params):
    vq_type = params.vq_type
    if vq_type == "dq":
        from dq import DQAE as model
    elif vq_type == "vq":
        from dq import VQVAE as model
    else:
        raise Exception("Unrecognized type: %s" % vq_type)
    kwargs = {key: getattr(params, key, None) for key in model.MODEL_ARGUMENTS}

    kwargs["n_hier"] = [int(i) for i in eval("".join(kwargs["n_hier"]))]

    return model(**kwargs)


def load_dq(device, save_dir=None, save_path=None, verbose=True, use_amp=True):
    from torch import optim

    from utils import load_chkpt_dict

    ckpt_dict, load_path = load_chkpt_dict(save_dir, save_path)

    config = ckpt_dict["model_config"]

    stats = ckpt_dict["stats"]
    model_state = ckpt_dict["model"]

    model = create_dq(config)
    model.load_state_dict(model_state,)
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(ckpt_dict["optimizer"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    cur_epoch = stats["cur_epoch"]
    iters = stats["iters"]
    best_loss = stats["best_loss"]
    loss = stats["train_loss"]
    if verbose:
        print("Current Epoch: %s Iteration: %s" % (cur_epoch, iters))
        print(f"Current Loss: {loss:.2e} Best Loss: {best_loss:.2e}")
        print("Loaded checkpoint: %s" % load_path)
        print("Model Config: %s" % config)
    return model, optimizer, scaler, config, stats


def save_dq(model, optimizer, scaler, config, stats, save_file):
    save_dict = {
        "model": model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict(),
        "scaler": scaler,
        "stats": stats,
        "model_config": config,
        "optimizer": optimizer.state_dict(),
    }

    torch.save(save_dict, save_file)
