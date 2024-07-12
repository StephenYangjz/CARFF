# CARFF Mixture Density Network

## Installation Instructions
Create a conda environment with Python 3.9
```
conda create --name mdn python=3.9
conda activate mdn
```

Install all dependencies
```
pip install -r requirements.txt
```

## Training the MDN
### Configuration setup
Optionally modify experiment and MDN model parameters in `mdn_config.yaml`. By default the code uses this path for the configuration file. To provide a new one use the `--config_path` flag.

Provide the `transforms.json` file containing the embeddings saved by the PC-VAE using the `--transform_path` flag and provide a path to save the MDN model checkpoint using the `--save_model_path` flag. Below is an example of how to run the code:
```
python run.py --transform_path ./transforms_val.json --save_model_path mdn_checkpoint
```

The code also saves a plot of the loss over epochs for the MDN model to ensure that the training performed correctly to `mdn_loss.png`.
