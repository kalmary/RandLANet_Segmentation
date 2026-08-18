import torch
import torch.nn as nn

import argparse
from tqdm import tqdm
import pathlib as pth
import sys


model_pipeline_dir = pth.Path(__file__).parent
src_dir = model_pipeline_dir.parent
sys.path.append(str(model_pipeline_dir))
sys.path.append(str(src_dir))

from RandLANet_CB import RandLANet
from _data_loader import make_loader

from utils import load_json, load_model, convert_str_values
from utils import calculate_accuracy, compute_mIoU
from utils import compute_pos_weights_prob, FocalLoss, get_intLabels, get_Probabilities
from utils import Plotter, ClassificationReport


def _eval_model(config_dict: dict,
                model: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    class_weights = compute_pos_weights_prob(
        data_dir=config_dict['data_path_test'],
        num_classes=config_dict['num_classes'],
        power=0.5,
    )
    test_loader, _ = make_loader(
        data_dir=config_dict['data_path_test'],
        num_points=config_dict['num_points'],
        batch_size=config_dict['batch_size'],
        query_workers=config_dict.get('query_workers', 7),
        shuffle=False,
    )

    output_batches = []
    label_batches = []

    model.eval()
    pbar = tqdm(test_loader, desc="Testing", unit="batch")
    with torch.no_grad():
        for batch_x, batch_y in pbar:
            outputs = model(batch_x.to(config_dict['device']))
            output_batches.append(outputs.cpu())
            label_batches.append(batch_y.cpu())

    if not output_batches:
        raise RuntimeError("Test loader produced no batches")

    return (
        torch.cat(output_batches, dim=0),
        torch.cat(label_batches, dim=0),
        class_weights,
    )


def calculate_metrics(outputs: torch.Tensor,
                      labels: torch.Tensor,
                      class_weights: torch.Tensor,
                      num_classes: int,
                      focal_loss_gamma: float) -> dict:
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=focal_loss_gamma,
    )
    loss = criterion(outputs, labels).item()
    accuracy = calculate_accuracy(outputs, labels)

    probabilities = get_Probabilities(outputs)
    predictions = get_intLabels(probabilities)
    miou, class_iou = compute_mIoU(predictions, labels, num_classes)

    probabilities = probabilities.movedim(1, -1).reshape(-1, num_classes)
    return {
        'loss': loss,
        'accuracy': accuracy,
        'miou': miou,
        'class_iou': class_iou.numpy(),
        'labels': labels.reshape(-1).numpy(),
        'probabilities': probabilities.numpy(),
        'predictions': predictions.reshape(-1).numpy(),
    }

def eval_model_front(config_dict: dict,
         model: nn.Module,
         paths: list[pth.Path]):

    model_path = paths[0]
    model_name = model_path.stem

    plot_dir = paths[1]
    plot_dir.mkdir(exist_ok=True, parents=True)

    outputs, labels, class_weights = _eval_model(
        config_dict=config_dict,
        model=model,
    )
    metrics = calculate_metrics(
        outputs=outputs,
        labels=labels,
        class_weights=class_weights,
        num_classes=config_dict['num_classes'],
        focal_loss_gamma=config_dict['focal_loss_gamma'],
    )
    del outputs, labels

    plotter = Plotter(config_dict['num_classes'], plots_dir=plot_dir)
    plotter.cnf_matrix(
        f'confusion_matrix_{model_name}.png',
        target=metrics['labels'],
        prediction=metrics['predictions'],
        num_classes=config_dict['num_classes'],
    )
    plotter.prc_curve(
        f'precision_recall_curve_{model_name}.png',
        target=metrics['labels'],
        pred_prob=metrics['probabilities'],
    )
    plotter.roc_curve(
        f'roc_curve_{model_name}.png',
        target=metrics['labels'],
        pred_prob=metrics['probabilities'],
    )
    
    print('='*20)
    print('MODEL TESTED')
    print('Model path', model_path)
    print('Loss: ', metrics['loss'])
    print('Accuracy: ', metrics['accuracy'])
    print('mIoU: ', metrics['miou'])
    print('IoU per class: ', metrics['class_iou'])
    print('Plots saved to:', plot_dir)
    print('='*20)
    
    metrics_report = (
        f"Loss: {metrics['loss']}\n"
        f"Accuracy: {metrics['accuracy']}\n"
        f"mIoU: {metrics['miou']}\n"
        f"IoU per class: {metrics['class_iou']}"
    )
    ClassificationReport(file_path=plot_dir.joinpath(f'classification_report_{model_name}.txt'),
                         pred=metrics['predictions'],
                         target=metrics['labels'],
                         additional_info=metrics_report)

def test_function(config_dict: dict,
                  model):
    val_loader, _ = make_loader(
        data_dir=config_dict['data_path_test'],
        num_points=config_dict['num_points'],
        batch_size=config_dict['batch_size'],
        query_workers=config_dict.get('query_workers', 7),
        shuffle=False,
    )
    
    batch_x, _ = next(iter(val_loader))
    batch_x = batch_x.to(config_dict['device'])

    model.eval()
    outputs = model(batch_x)

    expected_shape = (
        batch_x.shape[0],
        config_dict['num_classes'],
        batch_x.shape[1],
    )
    if outputs.shape == expected_shape:
        print('Model works as expected')
    else:
        print(f'Model does not work as expected\n')
        print(f'Expected output shape: {expected_shape}\nReceived: {outputs.shape}')

def parser(args=None):
        
    """
    Parse command-line arguments for model evaluation.
    Accepts model naming and evaluation mode selection.
    Returns parsed arguments with validation for device choices and formatted help text display.
    """
    
    parser = argparse.ArgumentParser(
        description="Script for testing the choosen model",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help=(
            "Base of the model's name.\n"
            "When iterating, name also gets an ID."
        )
    )

    parser.add_argument(
        '--mode',
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Evaluation mode.\n"
            'Pick:\n'
            '0: testing mode - check if model compiles and works as expected\n'
            '1: evaluate trained model'
        )
    )

    return parser.parse_args(args)

def main():
    args = parser()
    base_path = pth.Path(__file__).parent
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    device = torch.device('cuda')

    model_name = args.model_name
    model_name_no_num = model_name.rsplit('_', 1)[0]

    model_dir = base_path.joinpath(f'training_results/{model_name_no_num}')
    config_trained_dir = model_dir.joinpath('dict_files')
    model_path = config_trained_dir.joinpath(f'{model_name}_config.json')
    plot_dir = model_dir.joinpath('plots')

    config_dict = load_json(model_path)
    config_dict = convert_str_values(config_dict)
    config_dict['device'] = device
    
    model = RandLANet(model_config=config_dict['model_config'], n_classes=config_dict['num_classes'])
    model = load_model(file_path=model_dir.joinpath(f'{model_name}.pt'),
                       model=model,
                       device=device)
    model.eval()

    if args.mode == 0:
        test_function(config_dict, model)
    elif args.mode == 1:
        eval_model_front(config_dict=config_dict,
                         model=model,
                         paths=[model_path,
                                plot_dir])
        



if __name__ == '__main__':
    main()

