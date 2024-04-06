import os
import cv2
import numpy as np
import pandas as pd
import argparse
import torch
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_DEPTH = 0.001
MAX_DEPTH = 10.0


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def create_model(pretrained_resource):
    # Load default pretrained resource defined in config if not set
    overwrite = {"pretrained_resource": pretrained_resource}
    config = get_config("zoedepth", "eval", "pering", **overwrite)
    model = build_model(config)
    model = model.to(DEVICE)    
    return model

def model_infer(model, input_frame):
    depth = model.infer_pil(input_frame)
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    return depth

def run(
    output_path: str,
    model_filepath: str,
    eval_test: bool,
    save_images=False,
    ):
    """Evaluates a pretrained model using a specified test set.

    Args:
        output_path (str): path to output folder
        model_filepath (str): model weights .pt filepath
        eval_test (bool): option to evaluate on test set
        save_images (bool): save images to disk
    """   

    archives_path = '../dataset/{0}/{1}/data/{2}'
    
    if eval_test:
        filenames_path = "test_files.txt"
    else:
        filenames_path = "val_files.txt"
    with open(f'./splits/pering_complete/{filenames_path}') as f:
        archives = f.readlines()
    
    os.makedirs(output_path, exist_ok=True)
    model = create_model(model_filepath)

    num_images = len(archives)
    errors = list()
    for index, archive in enumerate(archives):
        directory, number, _ = archive.split()
        archive_name = '{:010d}.png'.format(int(number))

        input_path = os.path.join('../dataset', directory, 'cam0', 'data', archive_name)
        print("Processing {} ({}/{})".format(input_path, index + 1, num_images))

        input_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        prediction_depth = model_infer(model=model, input_frame=input_image)

        # output
        if output_path is not None:
            os.makedirs(os.path.join(output_path, directory), exist_ok=True)
            filename = os.path.join(output_path, directory, archive_name)
            if save_images:
                cv2.imwrite(filename + ".png", np.uint16((2**16 - 1) * (prediction_depth / MAX_DEPTH)) )

        gt_path = archives_path.format(directory, 'depth0', archive_name)
        gt_depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        # GT metric.
        gt_depth = gt_depth / (1000)
        mask = gt_depth > 0
        
        filename = os.path.join(output_path, directory, archive_name + '-gt')
        if save_images:
            cv2.imwrite(filename + ".png", np.uint16((2**16 - 1) * (gt_depth / MAX_DEPTH)))

        errors.append(compute_errors(gt_depth[mask], prediction_depth[mask]))

    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    # Log values as csv.
    errors_df = pd.DataFrame(
        data=[mean_errors.tolist()],
        columns=["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    )
    errors_df.to_csv(
        os.path.join(
            output_path,
            "results.csv"
        ),
        index=False
    )

    print("\n-> Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-o',
        '--output_path',
        default=None,
        help='Folder for output results'
    )
    parser.add_argument(
        '-p',
        '--pretrained_weights',
        help='model weights filepath'
    )
    parser.add_argument(
        "--eval_test",
        help="if set, use test_files.txt",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '-s',
        '--save_images',
        action='store_true',
        help='Save images to disk'
    )

    args = parser.parse_args()

    # compute depth maps
    run(
        output_path=args.output_path,
        model_filepath=args.pretrained_weights,
        eval_test=args.eval_test,
        save_images=args.save_images
    )
