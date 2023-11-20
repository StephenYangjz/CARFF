# CARFF: Conditional Auto-encoded Radiance Field for Forecasting

## Installation Instructions
```
# Create a conda environment with Python 3.9
conda create --name pc-vae python=3.9
conda activate pc-vae

# Install PyTorch according to official website
# For our system setup:
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install the rest of the dependencies
pip install -r requirements.txt
```

## Running Experiments
### Configuration setup
Modify experiment parameters and data path to point to local copy in `dec_cond_vae.yaml`:
```
data_path: "../carla_town1_data"
```
### Train PC-VAE on the dataset
```
# With constant KLD loss weight
python run.py -c configs/dec_cond_vae.yaml

# With KLD delayed linear scheduling
python run.py -c configs/dec_cond_vae.yaml --kld_scheduler True
```
