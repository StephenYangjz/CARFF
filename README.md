# CARFF: Conditional Auto-encoded Radiance Field for Forecasting
### _(Accepted to ECCV 2024)_


[Project Page With Video](https://stephenyangjz.github.io/carff_website)

This is an official implementation. The codebase is implemented using [PyTorch](https://pytorch.org/) and tested on [Ubuntu](https://ubuntu.com/) 22.04.2 LTS.

## Abstract
We propose CARFF, a method for predicting future 3D scenes given past observations, such as 2D egocentric images. Our method maps an image to a distribution over plausible latent scene configurations using a probabilistic encoder, and predicts the evolution of the hypothesized scenes through time. Our latent scene representation conditions a global Neural Radiance Field (NeRF) to represent a 3D scene model, which enables intuitive and explainable predictions. Unlike prior work in neural rendering, our approach models both uncertainty in environment state and dynamics. We employ a two-stage training of Pose-Conditional-VAE and NeRF to learn 3D representations. Additionally, we auto-regressively predict latent scene representations as a partially observable Markov decision process, utilizing a mixture density network. We demonstrate the utility of our method in realistic scenarios using the CARLA driving simulator, where CARFF can be used to enable efficient trajectory and contingency planning in complex multi-agent autonomous driving scenarios involving visual occlusions.

## Installation

### `Configure environment`

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge).

Please follow the readme files for each folder to create and activate a virtual environment for each subpart of this repository.

## Two-Stage Training and Inference

### CARFF's two stage training process

The PC-VAE encodes images into Gaussian latent distributions. Upper right: the pose-conditional decoder stochastically decodes the sampled latent using the given camera pose into an image. The decoded reconstruction and ground truth images are used in the MSE loss for the PC-VAE. Lower right: a NeRF is trained by conditioning on latent variables sampled from the optimized Gaussian parameters. These parameters characterize the timestamp distributions derived from PC-VAE. We use a separate MSE loss for the NeRF as well.

![train](./images/train.jpg)


### CARFF's auto-regressive inference pipeline

The input image is encoded using pre-trained PC-VAE to obtain a latent distribution, which is fed into our MDN model. The MDN predicts a mixture of Gaussians, which is sampled to obtain a predicted latent used to render a 3D view of the scene. To perform auto-regressive predictions, we probe the NeRF for the location of the car and feed the information back to the pre-trained encoder to predict the scene at the next timestamp.

![inference](./images/inference.jpg)

## Results
### Accuracy and recall curves from predicted beliefs

From the number of samples starting at 0 to 50, the belief state coverage generated under partial observation (recall), and the proportion of correct beliefs sampled under full observation (accuracy) is plotted for predicted beliefs. There is an ideal margin between the two as shown in the curves above for the two Multi-Scene CARLA datasets used to train our model.

<!-- ![recall_acc_curves](./images/recall_acc_curves.png) -->
<img src="./images/recall_acc_curves.png" alt="drawing" width="300"/>

### CARFF planning with controllers

CARFF-based controllers outperform baseline controllers by choosing the optimal action in potential collision scenarios over all 30 trials conducted.

<!-- ![controller](./images/controller.png, ) -->
<img src="./images/controller.png" alt="drawing" width="300"/>

## Related Works
![related](./images/related_works.png)

[[1] NeRF-VAE: A Geometry Aware 3D Scene Generative Model (ICML 2021)](https://arxiv.org/abs/2104.00587)\
[[2] 3D Neural Scene Representations for Visuomotor Control (CoRL 2021)](https://arxiv.org/abs/2107.04004)\
[[3] Vision-Only Robot Navigation in a Neural Radiance World](https://arxiv.org/abs/2110.00168)\
[[4] Is Anyone There? Learning a Planner Contingent on Perceptual Uncertainty (CoRL 2022)](https://openreview.net/forum?id=2CSj965d9O4)

## License and Citation
<!-- ```bibtex
@article{yang2023carff,
  title={{CARFF}: Conditional Auto-encoded Radiance Field for 3D Scene Forecasting},
  author={Yang, Jiezhi ``Stephen'' and Desai, Khushi and Bhatia, Harshil
  and Packer, Charles and Gonzalez, Joseph E.},
  journal={arXiv preprint arXiv},
  year={2023}
}
``` -->
```
@InProceedings{10.1007/978-3-031-73024-5_14,
author="Yang, Jiezhi
and Desai, Khushi
and Packer, Charles
and Bhatia, Harshil
and Rhinehart, Nicholas
and McAllister, Rowan
and Gonzalez, Joseph E.",
editor="Leonardis, Ale{\v{s}}
and Ricci, Elisa
and Roth, Stefan
and Russakovsky, Olga
and Sattler, Torsten
and Varol, G{\"u}l",
title="CARFF: Conditional Auto-Encoded Radiance Field for 3D Scene Forecasting",
booktitle="Computer Vision -- ECCV 2024",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="225--242",
abstract="We propose CARFF: Conditional Auto-encoded Radiance Field for 3D Scene Forecasting, a method for predicting future 3D scenes given past observations. Our method maps 2D ego-centric images to a distribution over plausible 3D latent scene configurations and predicts the evolution of hypothesized scenes through time. Our latents condition a global Neural Radiance Field (NeRF) to represent a 3D scene model, enabling explainable predictions and straightforward downstream planning. This approach models the world as a POMDP and considers complex scenarios of uncertainty in environmental states and dynamics. Specifically, we employ a two-stage training of Pose-Conditional-VAE and NeRF to learn 3D representations, and auto-regressively predict latent scene representations utilizing a mixture density network. We demonstrate the utility of our method in scenarios using the CARLA driving simulator, where CARFF enables efficient trajectory and contingency planning in complex multi-agent autonomous driving scenarios involving occlusions. Video and code are available at: www.carff.website.",
isbn="978-3-031-73024-5"
}
```

