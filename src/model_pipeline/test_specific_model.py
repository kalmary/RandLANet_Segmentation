import json
import torch
from torchinfo import summary
from RandLANet_CB import RandLANet


def model_info(config_path: str, n_classes: int, device: str = 'cpu'):
    config = json.load(open(config_path))
    model = RandLANet(model_config=config, n_classes=n_classes).to(device)
    dummy = torch.zeros(6, 16384, config['d_in'], device=device)
    summary(model, input_data=dummy, depth=3, col_names=["num_params", "trainable"])


if __name__ == '__main__':
    model_info('src/model_pipeline/model_configs/config_model_8.json', n_classes=10)