# https://discuss.streamlit.io/t/how-to-stream-ip-camera-in-webrtc-streamer/31379/4
import cv2
import copy
import gc
import numpy as np
import streamlit as st
from streamlit.runtime.scriptrunner import (
    add_script_run_ctx
)
from threading import Thread
import time
import torch

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAST_INPUT_FRAME = np.empty(shape=(1,1))
MODEL = None
UPPER_LIMIT_PERC = 0.3
UPPER_LIMIT = 0
LOWER_LIMIT_PERC = 0.2
LOWER_LIMIT = 0
ZOEDEPTH_MODEL_FILEPATH = "local::./results/ZoeDepth_demo.pt"

def create_model(pretrained_resource):
    # Load default pretrained resource defined in config if not set
    overwrite = {"pretrained_resource": pretrained_resource}
    config = get_config("zoedepth", "eval", "pering", **overwrite)
    model = build_model(config)
    model = model.to(DEVICE)    
    return model

def model_infer(model, input_frame):
    start_time = time.time()
    depth = model.infer_pil(input_frame)
    stop_time = time.time()
    
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    print('FPS =', 1 / (stop_time - start_time))

    return depth

def get_depth_view(depth):
    # Relative view.
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = depth * 255
    depth = depth.astype(np.uint8)
    return depth

def stop_inference_routine():
    global input_capture
    global input_viewer
    global output_viewer
    global MODEL
    # Frontend.
    input_viewer = st.image([])
    output_viewer = st.image([])
    input_capture.release()
    del input_capture
    # Model.
    del MODEL
    torch.cuda.empty_cache()
    gc.collect()
    MODEL = None
    quit()

def get_input_frame(video_capture):
    global LAST_INPUT_FRAME
    global run_button
    global input_viewer
    while run_button:
        status, frame = video_capture.read()
        if status:
            LAST_INPUT_FRAME = copy.deepcopy(frame)
            draw_roi_lines(frame)
            input_viewer.image(image=frame, channels="BGR")
    
    # Stop view.
    stop_inference_routine()

def get_roi(frame):
    global UPPER_LIMIT_PERC
    global UPPER_LIMIT
    global LOWER_LIMIT_PERC
    global LOWER_LIMIT
    # Business rule: robot limits.
    height, width  = frame.shape[0:2]
    UPPER_LIMIT = int(UPPER_LIMIT_PERC * height)
    LOWER_LIMIT = int((1 - LOWER_LIMIT_PERC) * height)

def draw_roi_lines(frame):
    global UPPER_LIMIT
    global LOWER_LIMIT
    get_roi(frame)
    # Business rule: robot limits.
    height, width  = frame.shape[0:2]
    # Upper limit.
    cv2.line(
        frame,
        (0        , UPPER_LIMIT),
        (width - 1, UPPER_LIMIT),
        (0, 0, 255),
        10
    )
    # Lower limit.
    cv2.line(
        frame,
        (0        , LOWER_LIMIT),
        (width - 1, LOWER_LIMIT),
        (0, 0, 255),
        10
    )

def get_nearest_in_roi(depth):
    global nearest_metric_text
    global UPPER_LIMIT
    global LOWER_LIMIT
    # Business rule.
    roi = depth[UPPER_LIMIT:LOWER_LIMIT, :]
    nearest_metric_text.write('Nearest distance in RoI: {:.2f} m'.format(roi.min()))

# Web application.
st.title("Video Object Depth Estimation")

run_button = st.checkbox('Generate depth', value=False)

input_capture = cv2.VideoCapture('rtsp://0.0.0.0:8554/input')
input_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# Video frame viewer.
column_1, column_2 = st.columns(2)
with column_1:
    input_viewer = st.image([])
with column_2:
    output_viewer = st.image([])
    nearest_metric_text = st.empty()

if run_button:
    MODEL = create_model(pretrained_resource=ZOEDEPTH_MODEL_FILEPATH)
    input_thread = Thread(target=get_input_frame, args=[input_capture])
    input_thread.daemon = True
    add_script_run_ctx(input_thread)
    input_thread.start()

while run_button:
    if len(LAST_INPUT_FRAME) > 1:
        depth_frame = model_infer(
            model=MODEL,
            input_frame=LAST_INPUT_FRAME
        )
        get_nearest_in_roi(depth_frame)
        depth_frame = get_depth_view(depth_frame)
        output_viewer.image(image=depth_frame)
