import argparse
from PIL import Image
from pprint import pprint
import torch
import os

from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import save_raw_16bit

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(config, cli_args):
    model = build_model(config)
    model = model.to(DEVICE)
    
    print(cli_args)
    os.makedirs(cli_args.output_folder, exist_ok=True)
    for root, dir, files in os.walk(cli_args.input_folder):
        for file in files:
            image_path = os.path.join(root, file)
            image = Image.open(image_path).convert("RGB")
            depth = model.infer_pil(image)
            
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = depth*256
            
            save_filepath = os.path.join(cli_args.output_folder, file)
            save_raw_16bit(depth, save_filepath)

def eval_model(model_name, pretrained_resource, cli_args, dataset='nyu', **kwargs):
    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    pprint(config)
    main(config, cli_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default=None, help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")
    parser.add_argument("-i", "--input_folder", type=str, required=False,
                        default='./inputs', help="Input images folder")
    parser.add_argument("-o", "--output_folder", type=str, required=False,
                        default='./outputs', help="Output images folder")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    if "ALL_INDOOR" in args.dataset:
        datasets = ALL_INDOOR
    elif "ALL_OUTDOOR" in args.dataset:
        datasets = ALL_OUTDOOR
    elif "ALL" in args.dataset:
        datasets = ALL_EVAL_DATASETS
    elif "," in args.dataset:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(
            args.model,
            pretrained_resource=args.pretrained_resource,
            dataset=dataset,
            cli_args=args,
            **overwrite_kwargs
        )
