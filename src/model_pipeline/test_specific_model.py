import argparse
import json
from pathlib import Path


def model_info(config_path: str, n_classes: int, device: str = 'cpu'):
    import torch
    from torchinfo import summary

    if __package__:
        from .randlanet_cb import Randlanet
    else:
        from randlanet_cb import Randlanet

    with open(config_path) as config_file:
        config = json.load(config_file)
    model = Randlanet(model_config=config, n_classes=n_classes).to(device)
    dummy = torch.zeros(6, 16384, config['d_in'], device=device)
    summary(model, input_data=dummy, depth=3, col_names=["num_params", "trainable"])


def argparser(argv=None):
    parser = argparse.ArgumentParser(description="Display a RandLANet model summary.")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = argparser(argv)
    model_info(args.config_path, n_classes=args.n_classes, device=args.device)


if __name__ == '__main__':
    main()
