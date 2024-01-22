# Monodepth2

Reference present in:

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)
>
> [ICCV 2019 (arXiv pdf)](https://arxiv.org/abs/1806.01260)

<p align="center">
  <img src="assets/teaser.gif" alt="example input output gif" width="600" />
</p>

Original license code is for non-commercial use; 
please see the [license file](LICENSE) for terms.

If you find the original work useful in your research please consider citing the paper:

```
@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
year = {2019}
}
```



## ⚙️ Setup

Original authors ran experiments with PyTorch 0.4.1, CUDA 9.1, Python 3.6.6 and Ubuntu 18.04.
They recommend to create a virtual environment with Python 3.6.6

Install local dependencies with:
```shell
conda create -n monodepth2 python=3.6.6
conda activate monodepth2
conda install pytorch=1.10.2 torchvision=0.2.1 -c pytorch
pip3 install tensorboardX==1.4
pip3 install tensorboard==2.10.1
pip3 install matplotlib==3.3.4
pip3 install scikit-image==0.17.2
apt-get install python-opencv -y
pip3 install opencv-python==3.3.1.11
```

A docker environment can be set up using the [Dockerfile](./Dockerfile)

<!-- We recommend using a [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) to avoid dependency conflicts.

We also recommend using `pillow-simd` instead of `pillow` for faster image preprocessing in the dataloaders. -->


## 🖼️ Prediction for a single image

You can predict scaled disparity for a single image with:

```shell
python test_simple.py \
  --image_path assets/test_image.jpg \
  --model_name mono_640x192 \
  --ext jpg
```

On its first will download the `mono_640x192` pretrained model (99MB) 
into the `models/` folder.
The authors provide the following  options for `--model_name`:

| `--model_name`          | Training modality | Imagenet pretrained? | Model resolution  | KITTI abs. rel. error |  delta < 1.25  |
|-------------------------|-------------------|--------------------------|-----------------|------|----------------|
| [`mono_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip)          | Mono              | Yes | 640 x 192                | 0.115                 | 0.877          |
| [`stereo_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip)        | Stereo            | Yes | 640 x 192                | 0.109                 | 0.864          |
| [`mono+stereo_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip)   | Mono + Stereo     | Yes | 640 x 192                | 0.106                 | 0.874          |
| [`mono_1024x320`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip)         | Mono              | Yes | 1024 x 320               | 0.115                 | 0.879          |
| [`stereo_1024x320`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip)       | Stereo            | Yes | 1024 x 320               | 0.107                 | 0.874          |
| [`mono+stereo_1024x320`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip)  | Mono + Stereo     | Yes | 1024 x 320               | 0.106                 | 0.876          |
| [`mono_no_pt_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip)          | Mono              | No | 640 x 192                | 0.132                 | 0.845          |
| [`stereo_no_pt_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip)        | Stereo            | No | 640 x 192                | 0.130                 | 0.831          |
| [`mono+stereo_no_pt_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip)   | Mono + Stereo     | No | 640 x 192                | 0.127                 | 0.836          |

## 💾 KITTI training data

Download the portion used for experiments with of 
[KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
cd kitti_data
unzip "*.zip"
cd ..
```

Useful informations:
- Use flag `--png` to sinalize `.png` extension
- Use flag `--data_path` to point to dataset root dir

**Splits**

The train/test/validation splits are defined in the `splits/` folder.

By default, the code will train a depth model using 
[Zhou's subset](https://github.com/tinghuiz/SfMLearner)
- [train_files](./splits/eigen_zhou/train_files.txt)
- [validation_files](./splits/eigen_zhou/val_files.txt)


The evaluation script defaults to [Eigen split](./splits/eigen/)

To choose different splits:
- Use flag `--split` pointing to the name of desired dir in `./splits`


**Custom dataset**

You can train on a custom monocular or stereo dataset by writing 
a new dataloader class which inherits from `MonoDataset` 
– see the `KITTIDataset` class in `datasets/kitti_dataset.py` for an example.


## 📊 KITTI evaluation

To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py \
  --data_path ../dataset \
  --split pering_deer \
  --min_depth 0.001 \
  --max_depth 1.0 \
  --save_images
```
...assuming that you have placed the dataset in `../dataset/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model`:
```shell
python evaluate_depth.py \
  --load_weights_folder ./results/models/weights_19/ \
  --model_name experiment_name \
  --eval_split pering_deer \
  --data_path ../dataset \
  --eval_out_dir ./evaluate \
  --num_workers 4 \
  --batch_size 1 \
  --min_depth 0.001 \
  --max_depth 1.0 \
  --save_pred_disps \
  --save_pred_images \
  --eval_mono
```

For stereo models, you must use the `--eval_stereo` flag (see note below):


## ⏳ Training

By default models and tensorboard event files are saved to `./results`.
This can be changed with the `--log_dir` flag.


**Monocular training (Scratch):**
```shell
python train.py \
  --model_name experiment_name \
  --log_dir ./results \
  --split pering_deer \
  --data_path ../dataset \
  --dataset pering \
  --min_depth 0.001 \
  --max_depth 1.0 \
  --num_workers 4 \
  --batch_size 1 \
  --log_frequency 50 \
  --save_frequency 1 \
  --num_epochs 20 \
  --png \
  --no_cuda
```


### 💽 Finetuning a pretrained model

Load an existing model for finetuning:
```shell
python train.py \
  --model_name experiment_name \
  --load_weights_folder ./models/mono_640x192 \
  --log_dir ./results \
  --split pering_deer \
  --data_path ../dataset \
  --dataset pering \
  --min_depth 0.001 \
  --max_depth 1.0 \
  --num_workers 4 \
  --batch_size 1 \
  --log_frequency 50 \
  --save_frequency 1 \
  --num_epochs 20 \
  --png \
  --no_cuda
```


## 📦 Precomputed results

You can download the original authors precomputed 
disparity predictions from the following links:


| Training modality | Input size  | `.npy` filesize | Eigen disparities                                                                             |
|-------------------|-------------|-----------------|-----------------------------------------------------------------------------------------------|
| Mono              | 640 x 192   | 343 MB          | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192_eigen.npy)           |
| Stereo            | 640 x 192   | 343 MB          | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192_eigen.npy)         |
| Mono + Stereo     | 640 x 192   | 343 MB          | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192_eigen.npy)  |
| Mono              | 1024 x 320  | 914 MB          | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320_eigen.npy)          |
| Stereo            | 1024 x 320  | 914 MB          | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320_eigen.npy)        |
| Mono + Stereo     | 1024 x 320  | 914 MB          | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320_eigen.npy) |



## 👩‍⚖️ License
Copyright © Niantic, Inc. 2019. Patent Pending.
All rights reserved.
Please see the [license file](LICENSE) for terms.
