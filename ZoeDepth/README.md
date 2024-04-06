# **ZoeDepth: Combining relative and metric depth** (Official implementation)  <!-- omit in toc -->
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isl-org/ZoeDepth)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/shariqfarooq/ZoeDepth)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![PyTorch](https://img.shields.io/badge/PyTorch_v1.10.1-EE4C2C?&logo=pytorch&logoColor=white) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoedepth-zero-shot-transfer-by-combining/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=zoedepth-zero-shot-transfer-by-combining)

>#### [ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/abs/2302.12288)
> ##### [Shariq Farooq Bhat](https://shariqfarooq123.github.io), [Reiner Birkl](https://www.researchgate.net/profile/Reiner-Birkl), [Diana Wofk](https://dwofk.github.io/), [Peter Wonka](http://peterwonka.net/), [Matthias Müller](https://matthias.pw/)

[[Paper]](https://arxiv.org/abs/2302.12288)

## **Environment setup**
The project depends on :
- [pytorch](https://pytorch.org/) (Main framework)
- [timm](https://timm.fast.ai/)  (Backbone helper for MiDaS)
- pillow, matplotlib, scipy, h5py, opencv (utilities)

Install environment using `environment.yml` : 
```bash
conda env create -n zoe --file environment.yml
conda activate zoe
```

## Model files
Models are defined under `models/` folder, with `models/<model_name>_<version>.py` containing model definitions and  `models/config_<model_name>.json` containing configuration.

Single metric head models (Zoe_N and Zoe_K from the paper) have the common definition and are defined under `models/zoedepth` while as the multi-headed model (Zoe_NK) is defined under `models/zoedepth_nk`.

## **Training**
1. Adjust dataset paths and min/max depth in [config.py](./zoedepth/utils/config.py),
inside `DATASETS_CONFIG`.

2. Adjust model type and input size in `model` section inside
  [config_zoedepth.json](./zoedepth/models/zoedepth/config_zoedepth.json)

3. Adjust training hyper-parameters in `train` section inside
  [config_zoedepth.json](./zoedepth/models/zoedepth/config_zoedepth.json)

4. Train model (saved in WandB):
    ```bash
    python train_mono.py \
      -m zoedepth \
      -p="" \
      -d pering
    ```

## **Running model**
Run trained model with images in `./inputs/` being saved in `./outputs/`:
```bash
python run_custom.py \
  -m zoedepth \
  -p "local::./results/model_trained.pt" \
  -d pering \
  -i ./inputs/ \
  -o ./outputs/
```

## Streamlit WEB demo

1. Use a RTSP server to simulate a camera stream input in `rtsp://localhost:8554/input`.
  Using (mediamtx)[https://github.com/bluenviron/mediamtx/releases/tag/v1.6.0]:
    ```bash
    ./mediamtx
    ```

2. Publish to server:
    ```bash
    ffmpeg -re -stream_loop -1 -i /path/to/video.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://localhost:8554/input
    ```

3. Provide ZoeDepth model to be used in: `./results/ZoeDepth_demo.pt`

4. Run web demo:
    ```bash
    streamlit run demo.py
    ```

## **Evaluation**

```bash
python3 evaluate_depth.py \
  -o ./evaluate/experiment \
  -p local::./results/model_trained.pt \
  [--save_images] \
  [--eval_test]
```

## **Citation**
```
@misc{https://doi.org/10.48550/arxiv.2302.12288,
  doi = {10.48550/ARXIV.2302.12288},
  
  url = {https://arxiv.org/abs/2302.12288},
  
  author = {Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and Müller, Matthias},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```













