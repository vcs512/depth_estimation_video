import os
import cv2
import numpy as np
import pandas as pd
import argparse
from skimage import exposure


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


def disp_to_depth(disp, min_depth = 0.001, max_depth=1.0):
    """Convert network's disparity output into depth prediction.
    """
    
    # scaled_disp = (disp - disp.min()) / (disp.max() - disp.min())
    scaled_depth = 1.0 - disp
    depth = min_depth + ( max_depth - min_depth ) * scaled_depth
    
    depth = exposure.adjust_gamma(image=depth, gamma=5.0)
 
    return None, depth

def run(
    input_path,
    output_path,
    side=False,
    ):
    """Evaluates a pretrained model using a specified test set.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        side (bool): RGB and depth side by side in output images?
    """   

    archives_path = '../dataset/{0}/{1}/data/{2}'
    with open('./splits/pering_complete/val_files.txt') as f:
        archives = f.readlines()
    
    os.makedirs(output_path, exist_ok=True)

    num_images = len(archives)
    errors = list()
    for index, archive in enumerate(archives):
        directory, number, _ = archive.split()
        archive_name = '{:010d}.png'.format(int(number))

        disp_path = os.path.join('./outputs', directory, archive_name)
        print("Processing {} ({}/{})".format(disp_path, index + 1, num_images))

        prediction = cv2.imread(disp_path, cv2.IMREAD_GRAYSCALE)
        prediction = prediction / 255
        
        _, prediction_depth = disp_to_depth(
            disp=prediction,
            min_depth=0.001,
            max_depth=1.0
        )

        # output
        if output_path is not None:
            os.makedirs(os.path.join(output_path, directory), exist_ok=True)
            filename = os.path.join(
                output_path, directory, archive_name
            )
            if not side:
                cv2.imwrite(filename + ".png", np.uint16((2**16 - 1)*prediction_depth) )

        gt_path = archives_path.format(directory, 'depth0', archive_name)
        gt_depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        
        # # GT metric.
        # gt_depth = gt_depth / (2**16 - 1)

        # GT relative.
        gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())
        gt_depth = gt_depth / gt_depth.max()
        
        mask = gt_depth > 0
        
        filename = os.path.join(
                output_path, directory, archive_name + '-gt'
            )
        if not side:
            cv2.imwrite(filename + ".png", np.uint16((2**16 - 1)*gt_depth))

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

    parser.add_argument('-i', '--input_path',
                        default=None,
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)'
                        )

    parser.add_argument('-o', '--output_path',
                        default=None,
                        help='Folder for output images'
                        )

    parser.add_argument('-s', '--side',
                        action='store_true',
                        help='Output images contain RGB and depth images side by side'
                        )

    args = parser.parse_args()

    # compute depth maps
    run(args.input_path, args.output_path, args.side)
