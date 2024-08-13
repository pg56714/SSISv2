# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import skimage.io as io
import numpy as np

# from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg
import torch

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/MS_R_101_BiFPN_SSISv2_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    # parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    # parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", default="./", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        default="./res_masks/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    cfg = setup_cfg(args)
    args.input = [os.path.join(args.input, path) for path in os.listdir(args.input)]
    demo = VisualizationDemo(cfg)

    for path in tqdm.tqdm(args.input, disable=not args.output):
        img = cv2.imread(path)
        start_time = time.time()
        with torch.no_grad():
            instances, _ = demo.run_on_image(img)
        logger.info("{}: detected {} instances in {:.2f}s".format(
            path, len(instances), time.time() - start_time))

        # Generate mask image
        masks = instances.pred_masks
        mask_image = np.zeros(img.shape[:2], dtype=np.uint8)
        for idx, mask in enumerate(masks):
            # mask_image[mask.astype(bool)] = (idx + 1) * 20  # Use mask as a boolean index
            mask_image[mask.astype(bool)] = 255

        if args.output:
            out_filename = os.path.join(args.output, os.path.basename(path))
            cv2.imwrite(out_filename, mask_image)
        else:
            cv2.imshow(WINDOW_NAME, mask_image)
            if cv2.waitKey(0) == 27:
                break  # esc to quit

