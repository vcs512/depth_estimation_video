# Reference: https://github.com/isl-org/ZoeDepth/issues/10
import cv2
import numpy as np
from threading import Thread
import time
import torch

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAST_INPUT_FRAME = np.empty(shape=(1,1))
MODEL = None
UPPER_LIMIT_PERC = 0.3
LOWER_LIMIT_PERC = 0.2


def model_infer(model, input_frame):
    start_time = time.time()
    depth = model.infer_pil(input_frame)
    stop_time = time.time()

    print('FPS = {:.2f}'.format(1 / (stop_time - start_time)))

    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()

    return depth

def create_model(pretrained_resource):
    # Load default pretrained resource defined in config if not set
    overwrite = {"pretrained_resource": pretrained_resource}
    config = get_config("zoedepth", "eval", "pering", **overwrite)
    model = build_model(config)
    model = model.to(DEVICE)
    
    return model

input_capture = cv2.VideoCapture('rtsp://0.0.0.0:8554/input')
input_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
input_capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)

def get_input_frame(video_capture):
    global LAST_INPUT_FRAME
    while True:
        status, frame = video_capture.read()
        if status:
            LAST_INPUT_FRAME = frame
    
MODEL = create_model(
pretrained_resource="local::./results/ZoeDepth_swin2T_pering_depth10_MiDaStrain_nbins32_batch2_ephocs20v1_19-Mar_12-56-fb09484e3218_best.pt"
)
input_thread = Thread(target=get_input_frame, args=[input_capture])
input_thread.daemon = True
input_thread.start()

while True:
    if len(LAST_INPUT_FRAME) > 1:
        # Infer.
        depth_frame = model_infer(
            model=MODEL,
            input_frame=LAST_INPUT_FRAME
        )

        # Business rule: robot limits.
        height, width  = depth_frame.shape[0:2]
        # Upper limit.
        upper_limit = int(UPPER_LIMIT_PERC * height)
        cv2.line(
            LAST_INPUT_FRAME,
            (0        , upper_limit),
            (width - 1, upper_limit),
            (0, 0, 255),
            10
        )
        # Lower limit.
        lower_limit = int((1 - LOWER_LIMIT_PERC) * height)
        cv2.line(
            LAST_INPUT_FRAME,
            (0        , lower_limit),
            (width - 1, lower_limit),
            (0, 0, 255),
            10
        )

        # Process inside RoI.        
        roi = depth_frame[upper_limit:lower_limit, :]
        print('Nearest depth = {:.2f} m'.format(roi.min()))

        # Save debug images.
        cv2.imwrite('./debug.png', LAST_INPUT_FRAME)
