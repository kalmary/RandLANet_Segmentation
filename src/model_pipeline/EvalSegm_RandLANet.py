import numpy as np
import torch
from torchinfo import summary
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import pathlib as pth

from _data_loader import Dataset_RandLANet

import os
import sys

final_files_dir = '/home/michal-siniarski/Dokumenty/PROGRAMMING/LAS/final_files'
sys.path.append(final_files_dir)

from RandLANet_CB import RandLANet

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from utils.nn_utils.src.evaluation_plot_tools import Plotter, ClassificationReport

# from utils_universal.load_save_files import load_json, load_model
from utils.nn_utils.src.accuracy_metrics import calculate_accuracy, compute_mIoU, get_Probabilities, \
    get_intLabels, calculate_accuracy_weighted, calculate_class_weights, get_dataset_len, FocalLoss, DiceLoss, \
    compute_mIoU_from_flat
from tqdm import tqdm

def dummy_data4warmup(input_dim: int, n_classes: int):
    # dummy warmup
    dummy_input = torch.randn(10, 8192, input_dim).to(torch.device('cuda'))  # xyz + initial features
    y = torch.randint(0, n_classes, (8192,)).long()  # Example semantic labels

    return dummy_input, y

def eval_model():
    print('--- TESTING MODEL ---')

    model_name = 'RandLANetV4_11'

    base_path = pth.Path(__file__).parent

    plots_path = 'evaluation_plots'

    params_dict = load_json(f'{model_name}_config.json',
                            folder_name='model_configs',
                            base_path=base_path)

    if 'cuda' in params_dict['device'] and torch.cuda.is_available():
        dvc = 'cuda'
    else:
        dvc = 'cpu'
    device = torch.device(dvc)

    print(f'\nUtilizing {params_dict['device']} device.')


    test_dataset = Dataset_RandLANet(path_dir=params_dict['data_path_test'],
                                     num_points=8192,
                                    batch_size = 10,
                                    shuffle=False,
                                    device=torch.device('cpu'))

    testLoader = DataLoader(test_dataset,
                           batch_size=None,
                           num_workers=14,
                           pin_memory=False)


    model = RandLANet4(
        d_in = params_dict['input_dim'],
        num_classes=params_dict['num_classes'],
        num_neighbors=params_dict['k_model'],
        decimation=params_dict['decimation'],
        margin=0.2,
        distance_type='euclidean'
    ).to(device)

    model = load_model(model_name, model, base_path=base_path, folder_name='trained_models')
    model.to(device)

    total = get_dataset_len(testLoader)
    class_weights = calculate_class_weights(testLoader, params_dict['num_classes'], total, device=device)

    alpha_f = 0.6
    alpha_d = 0.1

    #warmup
    data, label = dummy_data4warmup(params_dict['input_dim'], params_dict['num_classes'])
    data = data.to(device)

    model.eval()
    _ = model(data)
    del data

    epoch_loss = 0.
    epoch_accuracy = 0.
    epoch_miou = 0.
    epoch_IoU_pc = torch.zeros((1, int(params_dict['num_classes'])), dtype=torch.float32)
    epoch_samples = 0

    all_predictions = []
    all_probs = np.zeros((0, params_dict['num_classes']))
    all_outputs = torch.zeros((0, params_dict['num_classes'], 8192))
    all_labels = []

    progress = tqdm(enumerate(testLoader), f"Testing", total=total)

    with (torch.no_grad()):
        for i, (batch_x, batch_y) in progress:

            model.eval()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # outputs, triplet_loss = model(batch_x, batch_y)
            outputs = model(batch_x)

            outputs = outputs.cpu()
            batch_y = batch_y.cpu()

            loss_v = (criterion_ft(outputs, batch_y) * alpha_f
                      + criterion_dt(outputs, batch_y) * alpha_d)
            
            accuracy_v = calculate_accuracy_weighted(outputs, batch_y, params_dict['num_classes'])

            epoch_loss += loss_v.item() * batch_y.size(0)
            epoch_accuracy += accuracy_v * batch_y.size(0)
            # epoch_miou += miou_v * batch_y.size(0)

            # epoch_IoU_pc = epoch_IoU_pc + IoU_pc * batch_y.size(0)
            epoch_samples += batch_y.size(0)

            avg_loss = epoch_loss / epoch_samples
            avg_accuracy = epoch_accuracy / epoch_samples
            # avg_miou = epoch_miou / epoch_samples
            # avg_iou_pc = epoch_IoU_pc / epoch_samples

            progress.set_postfix({
                "Loss": f"{avg_loss:.6f}",
                "Acc": f"{avg_accuracy:.6f}"
            })

            probs = get_Probabilities(outputs)
            predictions = get_intLabels(probs)


            probs = probs.numpy()
            predictions = predictions.numpy()

            # points = batch_x[0, :, :3].cpu().numpy()
            #
            # pcd0 = o3d.geometry.PointCloud()
            # pcd0.points = o3d.utility.Vector3dVector(points)
            # pcd0.paint_uniform_color([0.6, 0.6, 0.6])
            # pcd0.paint_uniform_color([0.6, 0.6, 0.6])
            # print(np.unique(predictions[0, :]))
            # for label in np.unique(predictions[0, :]):
            #     cluster = points[label == predictions[0, :]]
            #     print(label)
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(cluster)
            #     pcd.paint_uniform_color([1.0, 0., 0.])
            #     o3d.visualization.draw_geometries([pcd0, pcd])




            all_outputs = torch.concatenate([all_outputs, outputs.cpu()], dim=0)
            all_predictions.extend(predictions.flatten().tolist())

            all_probs = np.concatenate([all_probs, probs.reshape(-1, params_dict['num_classes'])], axis = 0)
            all_labels.extend(batch_y.numpy().flatten().tolist())


    miou_v, avg_iou_pc = compute_mIoU_from_flat(torch.asarray(all_predictions), torch.asarray(all_labels), params_dict['num_classes'])
    print('miou, iou: ', miou_v, avg_iou_pc)

    plotter_obj = Plotter(class_num=params_dict['num_classes'], model_name=model_name, base_path=plots_path)
    plotter_obj.cnf_matrix(f'{model_name}_CNF_MATRIX.png', np.asarray(all_labels), np.asarray(all_predictions))

    ClassificationReport(f'{model_name}_classification_report.txt',
                         np.asarray(all_predictions),
                         np.asarray(all_labels),
                         model_name=model_name,
                         mIoU=miou_v, IoU_pc=avg_iou_pc.tolist())



def main():
    eval_model()

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Set the start method to 'spawn'
    main()