import os
import cv2
import skimage
import numpy as np
import argparse
import pandas as pd
import torch

from midas.model_loader import default_models, load_model
import utils

from pprint import pprint

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
first_execution = True


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
    
    scaled_disp = (disp - disp.min()) / (disp.max() - disp.min())
    scaled_depth = 1.0 - scaled_disp
    depth = min_depth + ( max_depth - min_depth ) * scaled_depth
 
    return scaled_disp, depth

def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction

def run(
    input_path,
    output_path,
    model_path,
    model_type="dpt_beit_large_512",
    optimize=False,
    side=False,
    height=None,
    square=False,
    grayscale=True):
    """Evaluates a pretrained model using a specified test set.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
        model_type (str): the model type
        optimize (bool): optimize the model to half-floats on CUDA?
        side (bool): RGB and depth side by side in output images?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?
        grayscale (bool): use a grayscale colormap?
    """   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    archives_path = '../dataset/{0}/{1}/data/{2}'
    with open('./splits/pering_complete/val_files.txt') as f:
        archives = f.readlines()
    
    os.makedirs(output_path, exist_ok=True)

    num_images = len(archives)
    errors = list()
    for index, archive in enumerate(archives):
        directory, number, _ = archive.split()
        archive_name = '{:010d}.png'.format(int(number))
        
        input_path = archives_path.format(
            directory,
            'cam0',
            archive_name
        )
        print(input_path)
        
        print("Processing {} ({}/{})".format(input_path, index + 1, num_images))

        # input
        original_image_rgb = utils.read_image(input_path)
        image = transform({"image": original_image_rgb})["image"]

        # compute
        with torch.no_grad():
            prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
                                    optimize, False)

        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        _, prediction_depth = disp_to_depth(
            disp=prediction,
            min_depth=0.001,
            max_depth=1.0
        )

        # output
        if output_path is not None:
            os.makedirs(os.path.join(output_path, directory), exist_ok=True)
            filename = os.path.join(
                output_path, directory, archive_name + '-' + model_type
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
                output_path, directory, archive_name + '-gt-' + model_type
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
            "results_MODEL_{}.csv".format(model_type)
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

    parser.add_argument('-m', '--model_weights',
                        default=None,
                        help='Path to the trained weights of model'
                        )

    parser.add_argument('-t', '--model_type',
                        default='dpt_beit_large_512',
                        help='Model type: '
                             'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, '
                             'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, '
                             'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or '
                             'openvino_midas_v21_small_256'
                        )

    parser.add_argument('-s', '--side',
                        action='store_true',
                        help='Output images contain RGB and depth images side by side'
                        )

    parser.add_argument('--optimize', dest='optimize', action='store_true', help='Use half-float optimization')
    parser.set_defaults(optimize=False)

    parser.add_argument('--height',
                        type=int, default=None,
                        help='Preferred height of images feed into the encoder during inference. Note that the '
                             'preferred height may differ from the actual height, because an alignment to multiples of '
                             '32 takes place. Many models support only the height chosen during training, which is '
                             'used automatically if this parameter is not set.'
                        )
    parser.add_argument('--square',
                        action='store_true',
                        help='Option to resize images to a square resolution by changing their widths when images are '
                             'fed into the encoder during inference. If this parameter is not set, the aspect ratio of '
                             'images is tried to be preserved if supported by the model.'
                        )
    parser.add_argument('--grayscale',
                        action='store_true',
                        help='Use a grayscale colormap instead of the inferno one. Although the inferno colormap, '
                             'which is used by default, is better for visibility, it does not allow storing 16-bit '
                             'depth values in PNGs but only 8-bit ones due to the precision limitation of this '
                             'colormap.'
                        )

    args = parser.parse_args()


    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize, args.side, args.height,
        args.square, True)
