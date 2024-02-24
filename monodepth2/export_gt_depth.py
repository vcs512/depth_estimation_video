# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil
import cv2

from utils import readlines
from kitti_utils import generate_depth_map


def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True)
    parser.add_argument('--save_images',
                        help='save depth images',
                        action="store_true")
    parser.add_argument("--min_depth",
                        type=float,
                        help="minimum depth",
                        default=0.001)
    parser.add_argument("--max_depth",
                        type=float,
                        help="maximum depth",
                        default=80.0)
    parser.add_argument("--eval_test",
                        help="if set, use test_files.txt",
                        action="store_true")
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)

    if opt.eval_test:
        filenames_path = "test_files.txt"
    else:
        filenames_path = "val_files.txt"
    lines = readlines(os.path.join(split_folder, filenames_path))

    print("Exporting ground truth depths for {}".format(opt.split))
    
    if opt.save_images:
        save_dir = os.path.join(split_folder, "gt_depths")
        os.makedirs(save_dir)

    gt_depths = []
    for line in lines:

        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        if opt.split == "eigen" \
            or opt.split == "kitti_custom":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256
        
        elif opt.split.find('pering') != -1:
            # Relative depth.
            gt_depth_path = os.path.join(opt.data_path, folder, "depth0",
                                         "data", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32)
            gt_depth = gt_depth / 2**16
            gt_depth = gt_depth * (opt.max_depth - opt.min_depth) + opt.min_depth

        if opt.save_images:
            save_filepath = os.path.join(save_dir, "{:010d}.png".format(frame_id))
            relative_depth = np.clip(gt_depth, a_min=opt.min_depth, a_max=opt.max_depth)
            relative_depth = (relative_depth - opt.min_depth) / (opt.max_depth - opt.min_depth)
            relative_depth = np.uint16(relative_depth * (2**16 - 1))
            cv2.imwrite(save_filepath, relative_depth)

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_kitti()
