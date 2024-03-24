# https://discuss.streamlit.io/t/how-to-stream-ip-camera-in-webrtc-streamer/31379/4
import cv2
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

def model_infer(model, input_frame):
    start_time = time.time()
    depth = model.infer_pil(input_frame)
    stop_time = time.time()
    
    print('FPS =', 1 / (stop_time - start_time))
    
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = depth * 255
    depth = depth.astype(np.uint8)
    
    return depth

def create_model(pretrained_resource):
    # Load default pretrained resource defined in config if not set
    overwrite = {"pretrained_resource": pretrained_resource}
    config = get_config("zoedepth", "eval", "pering", **overwrite)
    model = build_model(config)
    model = model.to(DEVICE)
    
    return model

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
            LAST_INPUT_FRAME = frame
            input_viewer.image(image=LAST_INPUT_FRAME, channels="BGR")
    
    # Stop view.
    stop_inference_routine()

if run_button:
    MODEL = create_model(
    pretrained_resource="local::./results/ZoeDepth_swin2T_pering_depth10_MiDaStrain_nbins32_batch2_ephocs20v1_19-Mar_12-56-fb09484e3218_best.pt"
    )
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
        output_viewer.image(image=depth_frame)
