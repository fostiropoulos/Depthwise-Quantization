import argparse
import sys
from configparser import ConfigParser

import numpy as np
from tabulate import tabulate

from dataset import load_dataset
from train_dq import test_dq
from utils import TestConfig, load_dq

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
    test_config = TestConfig()
    test_config.load(**dict(config["test"]))

    device = flags.device
    trainloader, testloader = load_dataset(
        test_config.dataset,
        test_config.image_size,
        test_config.batch_size,
        drop_last=False,
    )

    model, _, _, model_config, _ = load_dq(
        device=device,
        save_dir=None,
        save_path=test_config.model_path,
        use_amp=test_config.use_amp,
    )

    print("Model Configuration: %s" % model_config)

    img, _ = next(iter(trainloader))

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
    model.eval()
    model = model.to(device)

    test_loss = test_dq(model, testloader, device)
    print(test_loss)
    sys.exit(1)
