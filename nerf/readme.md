# CARFF NeRF Decoder

This folder is the NeRF decoder implementation based on the pytorch implementation of [instant-ngp](https://github.com/NVlabs/instant-ngp), as described in [_Instant Neural Graphics Primitives with a Multiresolution Hash Encoding_](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf).

### TinyCUDANN requirements
Make sure to follow the requirements listed on the [TinyCUDANN](https://github.com/NVlabs/tiny-cuda-nn) repository before creating the conda environment to avoid running into issues during setup:
- A C++14 capable compiler. The following choices are recommended and have been tested:
    - Windows: Visual Studio 2019 or 2022
    - Linux: GCC/G++ 8 or higher
- A recent version of [CUDA](https://developer.nvidia.com/cuda-toolkit). The following choices are recommended and have been tested:
    - Windows: CUDA 11.5 or higher
    - Linux: CUDA 10.2 or higher
- Install the version of [PyTorch](https://pytorch.org/get-started/locally/) that matches your CUDA installation to avoid failing setup for TCNN
- [CMake](https://cmake.org/) v3.21 or higher.

**Linux**
Run the following command to install the following packages:
```
sudo apt-get install build-essential git
```
Additionally ensure that the CUDA installation is added to your path like the below example (replace 11.8 with the installed CUDA version):
```
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
```

**Windows**
To add CUDA to your path variable:
```
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set LD_LIBRARY_PATH=%LD_LIBRARY_PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib64
```

### Install with pip
Create a conda environment:
```
conda create --name torch-ngp python=3.9
conda activate torch-ngp
```
Install the requirements and TinyCUDANN:
```
# For CUDA 11.8 torch installation -- change this to your version of CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cmake --upgrade
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Raymarching import errors
If the error or similar errors detailed in this [GitHub issue](https://github.com/ashawkey/torch-ngp/issues/184) appears, navigate to `raymarching/backend.py` and change all instances of `c++14` to `c++17`.


# Usage

We use the same data format as instant-ngp, e.g., [armadillo](https://github.com/NVlabs/instant-ngp/blob/master/data/sdf/armadillo.obj) and [fox](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox). 
Please download and put them under `./data`.

First time running will take some time to compile the CUDA extensions.

```bash
python main_nerf.py path/to/data --workspace workspace_name
```


# Citation

If you find this work useful, a citation will be appreciated via:
```
@misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
}

@article{tang2022compressible,
    title = {Compressible-composable NeRF via Rank-residual Decomposition},
    author = {Tang, Jiaxiang and Chen, Xiaokang and Wang, Jingbo and Zeng, Gang},
    journal = {arXiv preprint arXiv:2205.14870},
    year = {2022}
}
```

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

