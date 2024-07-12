# CARFF Pose-Conditional Variarional Autoencoder

## Installation Instructions
Create a conda environment with Python 3.9
```
conda create --name pc-vae python=3.9
conda activate pc-vae
```

Install all dependencies
```
pip install -r requirements.txt
```

## Running Experiments
### Configuration setup
Modify experiment parameters and data path to point to local copy in `dec_cond_vae.yaml`:
```
data_path: "./data/carla_dataset"
```
Modify the log parameters to point to a directory to save training logs:
```
save_dir: "logs_carff_test/"
```
### Train PC-VAE on the dataset
By default, we have turned off the Weights & Biases flag. The code does include logging KLD, PSNR, loss, etc. to Weights & Biases if the `--wandb_toggle` flag is set to `True`. In order to use W&B logging, you will need to specify the entity name with the `--wandb_entity` flag and the project name with the `--wandb_project` flag. 
```
# With constant KLD loss weight
python run.py -c configs/dec_cond_vae.yaml

# With KLD delayed linear scheduling
python run.py -c configs/dec_cond_vae.yaml --kld_scheduler True
```

Example with W&B logging:
```
python run.py -c configs/dec_cond_vae.yaml --kld_scheduler True --wandb_toggle True --wandb_entity example_entity_name
```
