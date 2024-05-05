# <div align=""> Toward Practical Monocular Indoor Depth Estimation </div>


<a href="https://choyingw.github.io/">Cho-Ying Wu</a>, <a href="https://sites.google.com/view/jialiangwang/home">Jialiang Wang</a>, <a href="https://www.linkedin.com/in/michaelanthonyhall/">Michael Hall</a>, <a href="https://cgit.usc.edu/contact/ulrich-neumann/">Ulrich Neumann</a>, <a href="https://shuochsu.github.io/">Shuochen Su</a>

[<a href="https://arxiv.org/abs/2112.02306">arXiv</a>] [<a href="https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Toward_Practical_Monocular_Indoor_Depth_Estimation_CVPR_2022_paper.html">CVF open access</a>] [<a href="https://distdepth.github.io/">project site: data, supplementary</a>]

    
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/toward-practical-self-supervised-monocular/monocular-depth-estimation-on-nyu-depth-v2-4)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2-4?p=toward-practical-self-supervised-monocular)

## <div align="">Updates</div>
[Mar 2024]: Fix bugs for instability for coverting scale and shift 

[Jan 2024]: Add the online least square alignemnt for expert and student's scale and fix bugs for per-sample edge map calculation.

[Augest 2023]: Add a snippet for simple AR effects using z-buffer. See the last section

**[June 2023]: Revise the instruction for training codes and train on your own dataset.**

[June 2023]: Fix bugs in sample training code.

[June 2023]: Fix bugs in visualization and saving.

## <div align="">Introduction</div>


As this project includes data contribution, please refer to the project page for data download instructions, including SimSIN, UniSIN, and VA, as well as UniSIN leaderboard participation.

Advantage
<img src='fig/teaser.png'>

Results
<img src='fig/results_pc_1.png'>
<img src='fig/results_pc_2.png'>

### DistDepth
DistDepth is a highly robust monocular depth estimation approach for generic indoor scenes.
* Trained with stereo sequences without their groundtruth depth
* Structured and metric-accurate
* Run in an interactive rate with Laptop GPU
* Sim-to-real: trained on simulation and becomes transferrable to real scenes

## Docker setup

1. Build and start environment (GPU):
   ```bash
   docker compose up --build
   ```

2. Enter container:
   ```bash
   docker exec -it distdepth-distdepth-1 bash
   ```

## <div align=""> Inference Demo </div>

1. Download pretrained models 
[<a href="https://drive.google.com/file/d/1N3UAeSR5sa7KcMJAeKU961KUNBZ6vIgi/view?usp=sharing">here</a>]
(ResNet152, 246MB, illustation for averagely good in-the-wild indoor scenes).

2. Unzip the model under the directory './ckpts' containing the pretrained models.

3. Adjust desired images path in [demo_list.txt](./demo_list.txt)

4. Run inference inside the container:
   ```bash
   python3 demo.py
   ```

5. Results will be stored under `results/` dir.

Note that during inference, it is applied a 1.312 scale for models trained on SimSIN,
since SimSIN is created with stereo baseline of 13.12cm,
and during training it uses a stereo scale of 10cm.
(See [Issue 27](https://github.com/facebookresearch/DistDepth/issues/27#issue-1989386374))

## Example data

For a simple taste of training, download a smaller replica set 
[<a href="https://drive.google.com/file/d/1g-OXOsKeincRc1-O3x42wVRFKpog2aRe/view?usp=sharing">here</a>]

The folder structure should be

   ```
   .
   ├── SimSIN-simple
         ├── replica
         ├── replica_train.txt
   ```

Download weights

```shell
mkdir weights

wget -O weights/dpt_hybrid_nyu-2ce69ec7.pt https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_nyu-2ce69ec7.pt 
```

## Training

The below command trains networks by using current and past/future one frame:

```shell
python3 execute.py \
   --exe train \
   --model_name experiment_name \
   --frame_ids 0 -1 1 \
   --log_dir='./tmp' \
   --data_path /dataset \
   --dataset pering  \
   --batch_size 4 \
   --width 256 \
   --height 256 \
   --max_depth 10.0 \
   --min_depth 0.001 \
   --num_epochs 10 \
   --scheduler_step_size 8 \
   --learning_rate 0.0001 \
   --thre 0.95 \
   --num_layers 34 \
   --log_frequency 6 \
   --num_workers 0 \
   [--no_cuda]
```

Changing different expert network: See execute_func.py L59.

Switch to different version of DPTDepthModel.
The default now uses DPT finetuned on NYUv2
(NYUv2 model is more capable of indoor scenes, and midas model is for general purpose)

If you would like to use more frames, you'll need to leave more buffer frames in the data list file.
See below notes for details.


**Notes for training on your own dataset:**

1. Create your dataloader. You can find SimSIN sample (containing both temporal and stereo) under dataset/ , and then add your dataloader in execute_func.py L111.

2. In execute_func.py L130-141, add your data list file. See format in Replica sample data. Specifically each line contains <file_path> <space> <temporal_step> <step> <left_or_right_for_stereo>

3. Use the before commands to train on your data. Note that your data need to have stereo if you specify --use_stereo. If you sepcify frame_id -1, 1, you'll need to leave one buffer frame at the top and end to avoid reading from None. For example, replica sample data contain 0-49 time steps, but in the data list file, only 1-48 are in file  

## <div align=""> Evaluation</div>

SimSIN trained models, evaluation on VA

| Name | Arch | Expert | MAE | AbsRel | RMSE | acc@ 1.25 | acc@ 1.25^2 | acc@ 1.25^3 | Download |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DistDepth | ResNet152 | DPT Large | 0.252 | 0.175 | 0.371 | 75.1 | 93.9 | 98.4 | [model](https://drive.google.com/file/d/1X_VMg1LYLmm8xCloLRjqHtIslyXfOrn5/view?usp=sharing) |
| DistDepth | ResNet152 | DPT Legacy | 0.270 | 0.186 | 0.386 | 73.2 | 93.2 | 97.9 | [model](https://drive.google.com/file/d/1rTBSglo_h-Ke5HMe4xvHhCjpeBDRl6vx/view?usp=sharing) |
| DistDepth-Multi| ResNet101 | DPT Legacy | 0.243 | 0.169 | 0.362 | 77.1 | 93.7 | 97.9 | [model](https://drive.google.com/file/d/1Sg_dXAyKI2VfKzHiAu9i8WqT9I7Y9k0D/view?usp=sharing) |


Download VA (8G) first. Extract under the root folder.

      .
      ├── VA
            ├── camera_0
               ├── 00000000.png 
                   ......
            ├── camera_1
               ├── 00000000.png 
                   ......
            ├── gt_depth_rectify
               ├── cam0_frame0000.depth.pfm 
                   ......
            ├── VA_left_all.txt

Run   ``` bash eval.sh ```   The performances will be saved under the root folder.

To visualize the predicted depth maps in a minibatch (adjust batch_size for different numbers): 


```shell
python execute.py --exe eval_save --log_dir='./tmp' --data_path VA --dataset VA  --batch_size 10 --load_weights_folder <path to weights> --models_to_load encoder depth  --width 256 --height 256 --max_depth 10 --frame_ids 0 --num_layers 152
```

If missing 'weights/dpt_hybrid_nyu-2ce69ec7.pt' message pops up, download the model from [DPT](https://github.com/isl-org/DPT) and put it under 'weights'.

To visualize the predicted depth maps for all testing data on the list: 

```shell
python execute.py --exe eval_save_all --log_dir='./tmp' --data_path VA --dataset VA  --batch_size 1 --load_weights_folder <path to weights> --models_to_load encoder depth  --width 256 --height 256 --max_depth 10 --frame_ids 0 --num_layers 152
```

Only batch_size = 1 is valid under this mode.

Evaluation on NYUv2

Prepare <a href="https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html">NYUv2</a> data.

      .
      ├── NYUv2
            ├── img_val
               ├── 00001.png
               ......
            ├── depth_val
               ├── 00001.npy
               ......
               ......
            ├── NYUv2.txt

| Name | Arch | Expert | MAE | AbsRel | RMSE | acc@ 1.25 | acc@ 1.25^2 | acc@ 1.25^3 | Download |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DistDepth-finetuned | ResNet152 | DPT on NYUv2 | 0.308 | 0.113 | 0.444 | 87.3 | 97.3 | 99.3 | [model](https://drive.google.com/file/d/1kLJBuMOf0xSpYq7DtxnPpBTxMwW0ylGm/view?usp=sharing) |
| DistDepth-SimSIN | ResNet152 | DPT | 0.411 | 0.163 | 0.563 | 78.0 | 93.6 | 98.1 | [model](https://drive.google.com/file/d/1Hf_WPaBGMpPBFymCwmN8Xh1blXXZU1cd/view?usp=sharing) |

Change train_filenames (dummy) and val_filenames in execute_func.py to NYUv2. Then,

```shell
python execute.py --exe eval_measure --log_dir='./tmp' --data_path NYUv2 --dataset NYUv2  --batch_size 1 --load_weights_folder <path to weights> --models_to_load encoder depth  --width 256 --height 256 --max_depth 12 --frame_ids 0 --num_layers 152
```

## <div align="">Citation</div>

    @inproceedings{wu2022toward,
    title={Toward Practical Monocular Indoor Depth Estimation},
    author={Wu, Cho-Ying and Wang, Jialiang and Hall, Michael and Neumann, Ulrich and Su, Shuochen},
    booktitle={CVPR},
    year={2022}
    }

## License
DistDepth is CC-BY-NC licensed, as found in the LICENSE file.
