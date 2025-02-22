# Torch Neural Graphics Primatives - MSL Fork

This is a fork of the repository [torch-ngp](https://github.com/ashawkey/torch-ngp), which is itself based on [instant-ngp](https://github.com/NVlabs/instant-ngp) by Thomas Müller.

The goal of this repository is to provide an fast and easy to use implementation of basic NeRF utilities for more efficient research iteration.

* MSL verified timing results:
    - LEGO RESULTS
    - FOX RESULTS

## Installation
1) Clone this repository
    ```bash
    git clone --recursive git@github.com:StanfordMSL/torch-ngp.git
    ```
    * Make sure to use the recursive argument because of the `cutlass` submodule which will otherwise throw errors for some functionality.
    * If you want to add this repo as a submodule to a current project then use:
        ```bash
        git submodule add git@github.com:StanfordMSL/torch-ngp.git
        cd torch-ngp
        git submodule update --init --recursive # To add cutlass
        ```

2) Setup a Python environment 
    ```bash
    cd torch-ngp
    virtualenv venv # instructions for virtualenv but should be similar with conda etc.
    source venv/bin/activate
    pip install -r requirements.txt
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    
    # To install the torch_ngp package
    pip install -e .
    ```

3) Download the basic NeRF Datasets
    ```bash
    cd ... # wherever you want your data
    mkdir -p data
    cd data
    wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
    unzip nerf_example_data.zip
    cd ..
    ```

## Usage
There are a variety of ways to use this repository. The `main_nerf.py` script uses all of the functionality of the original repository to run NeRF examples. Find the usage instructions for this script in the documentation of [torch-ngp](https://github.com/ashawkey/torch-ngp).

Alternatively, `nerf_basic.py` shows a more stripped back implementation of the core functionality of the packages in this repository.

## Goals
    - [] Pose optimization functionality
    - [] Benchmark results

## Contents of old `readme.md` (click to expand)

<details>
<summary> CLICK </summary>

# torch-ngp

A pytorch implementation of [instant-ngp](https://github.com/NVlabs/instant-ngp), as described in [_Instant Neural Graphics Primitives with a Multiresolution Hash Encoding_](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf).

With the CUDA ray marching option for NeRF, for the fox dataset, we can:
* converge to a reasonable result in **~1min** (50 epochs). 
* render a 1920x1080 image in **~1s**. 

For the LEGO dataset, we can reach **~20FPS** at 800x800 due to efficient voxel pruning.

(Tested with a TITAN RTX. The speed is still 2-5x slower compared to the original implementation.)

**A GUI for training/visualizing NeRF is also available!**

https://user-images.githubusercontent.com/25863658/155265815-c608254f-2f00-4664-a39d-e00eae51ca59.mp4


# Progress

As the official pytorch extension [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) has been released, the following implementations can be used as modular alternatives. 
The performance and speed of these modules are guaranteed to be on-par, and we support using tinycudann as the backbone by the `--tcnn` flag.
Later development will be focused on reproducing the NeRF inference speed.

* Fully-fused MLP
    - [x] basic pytorch binding of the [original implementation](https://github.com/NVlabs/tiny-cuda-nn)
* HashGrid Encoder
    - [x] basic pytorch CUDA extension
    - [x] fp16 support 
* Experiments
    - SDF
        - [x] baseline
        - [ ] better SDF calculation (especially for non-watertight meshes)
    - NeRF
        - [x] baseline
        - [x] ray marching in CUDA.
* NeRF GUI
    - [x] supports training.
* Misc.
    - [x] improve rendering quality of cuda raymarching
    - [ ] improve speed (e.g., avoid the `cat` in NeRF forward)
    - [ ] support visualize/supervise normals (add rendering mode option).
    - [x] support blender dataset format.


# Install
```bash

git clone --recursive https://github.com/ashawkey/torch-ngp.git

cd torch-ngp

pip install -r requirements.txt

# (optional) install the tcnn backbone
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
Tested on Ubuntu with torch 1.10 & CUDA 11.3 on TITAN RTX.

Currently, `--ff` only supports GPUs with CUDA architecture `>= 70`.
For GPUs with lower architecture, `--tcnn` can still be used, but the speed will be slower compared to more recent GPUs.

# Usage

We use the same data format as instant-ngp, e.g., [armadillo](https://github.com/NVlabs/instant-ngp/blob/master/data/sdf/armadillo.obj) and [fox](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox). 
Please download and put them under `./data`.

First time running will take some time to compile the CUDA extensions.

```bash
# train with different backbones (with slower pytorch ray marching)
# for the colmap dataset, the default dataset setting `--mode colmap --bound 2 --scale 0.33` is used.
python main_nerf.py data/fox --workspace trial_nerf # fp32 mode
python main_nerf.py data/fox --workspace trial_nerf --fp16 # fp16 mode (pytorch amp)
python main_nerf.py data/fox --workspace trial_nerf --fp16 --ff # fp16 mode + FFMLP (this repo's implementation)
python main_nerf.py data/fox --workspace trial_nerf --fp16 --tcnn # fp16 mode + official tinycudann's encoder & MLP

# test mode
python main_nerf.py data/fox --workspace trial_nerf --fp16 --ff --test

# use CUDA to accelerate ray marching (much more faster!)
python main_nerf.py data/fox --workspace trial_nerf --fp16 --ff --cuda_ray # fp16 mode + FFMLP + cuda raymarching

# start a GUI for NeRF training & visualization
# always use with `--fp16 --ff/tcnn --cuda_ray` for an acceptable framerate!
python main_nerf.py data/fox --workspace trial_nerf --fp16 --ff --cuda_ray --gui

# test mode for GUI
python main_nerf.py data/fox --workspace trial_nerf --fp16 --ff --cuda_ray --gui --test

# for the blender dataset, you should add `--mode blender --bound 1.5 --scale 1.0`
# --mode specifies dataset type ('blender' or 'colmap')
# --bound means the scene is assumed to be inside box[-bound, bound]
# --scale adjusts the camera locaction to make sure it falls inside the above bounding box.
python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf --fp16 --ff --cuda_ray --mode blender --bound 1.5 --scale 1.0 
python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf --fp16 --ff --cuda_ray --mode blender --bound 1.5 --scale 1.0 --gui
```

check the `scripts` directory for more provided examples.

# Difference from the original implementation
* Instead of assuming the scene is bounded in the unit box `[0, 1]` and centered at `(0.5, 0.5, 0.5)`, this repo assumes **the scene is bounded in box `[-bound, bound]`, and centered at `(0, 0, 0)`**. Therefore, the functionality of `aabb_scale` is replaced by `bound` here.
* For the hashgrid encoder, this repo only implement the linear interpolation mode.
* For the voxel pruning in ray marching kernels, this repo doesn't implement the multi-scale density grid (check the `mip` keyword), and only use one `128x128x128` grid for simplicity. Instead of updating the grid every 16 steps, we update it every epoch, which may lead to slower first few epochs if using `--cuda_ray`.
* For the blender dataest, the default mode in instant-ngp is to load all data (train/val/test) for training. Instead, we only use the specified split to train in CMD mode for easy evaluation. However, for GUI mode, we follow instant-ngp and use all data to train (check `type='all'` for `NeRFDataset`).

# Update Logs
* 3.21: lots of modifications to improve PSNR, now we can reach ~33 for the LEGO dataset.
    * enhanced data provider (random sample rays from all training images, and pre-generate rays)
    * ported parts of TensoRF for comparison (not fully supported!).
    * known issue: pre-generating rays consumes much more CPU memory at starting. Shuffle of such a large dataset can be very slow. Dataloader needs more num_workers to keep the speed, but still sometimes unstable.
* 3.14: fixed the precision related issue for `fp16` mode, and it renders much better quality. Added PSNR metric for NeRF.
* 3.14: linearly scale `desired_resolution` with `bound` according to https://github.com/ashawkey/torch-ngp/issues/23.
    * known issue: very large bound (e.g., 16) leads to bad performance. Better to scale down the camera to fit into a smaller bounding box.
* 3.11: raymarching now supports supervising weights_sum (pixel alpha, or mask) directly, and bg_color is separated from CUDA to make it more flexible. Add an option to preload data into GPU.
* 3.9: add fov for gui.
* 3.1: add type='all' for blender dataset (load train + val + test data), which is the default behavior of instant-ngp.
* 2.28: density_grid now stores density on the voxel center (with randomness), instead of on the grid. This should improve the rendering quality, such as the black strips in the lego scene.
* 2.23: better support for the blender dataset.
* 2.22: add GUI for NeRF training.
* 2.21: add GUI for NeRF visualizing. 
    * known issue: noisy artefacts outside the camera covered region. It is related to `mark_untrained_density_grid` in instant-ngp.
* 2.20: cuda raymarching is finally stable now!
* 2.15: add the official [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) as an alternative backend.    
* 2.10: add cuda_ray, can train/infer faster, but performance is worse currently.
* 2.6: add support for RGBA image.
* 1.30: fixed atomicAdd() to use `__half2` in HashGrid Encoder's backward, now the training speed with fp16 is as expected!
* 1.29: 
    * finished an experimental binding of fully-fused MLP.
    * replace SHEncoder with a CUDA implementation.
* 1.26: add fp16 support for HashGrid Encoder (requires CUDA >= 10 and GPU ARCH >= 70 for now...).


# Acknowledgement

* Credits to [Thomas Müller](https://tom94.net/) for the amazing [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [instant-ngp](https://github.com/NVlabs/instant-ngp):
    ```
    @misc{tiny-cuda-nn,
        Author = {Thomas M\"uller},
        Year = {2021},
        Note = {https://github.com/nvlabs/tiny-cuda-nn},
        Title = {Tiny {CUDA} Neural Network Framework}
    }

    @article{mueller2022instant,
        title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
        author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
        journal = {arXiv:2201.05989},
        year = {2022},
        month = jan
    }
    ```

* The framework of NeRF is adapted from [nerf_pl](https://github.com/kwea123/nerf_pl):
    ```
    @misc{queianchen_nerf,
        author = {Quei-An, Chen},
        title = {Nerf_pl: a pytorch-lightning implementation of NeRF},
        url = {https://github.com/kwea123/nerf_pl/},
        year = {2020},
    }
    ```
* The NeRF GUI is developed with [DearPyGui](https://github.com/hoffstadt/DearPyGui).
</details>
