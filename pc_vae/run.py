import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin
from utils import NormalLogger

import wandb

os.environ['WANDB_SILENT']="true"

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help="Path to the config file",
                    default='configs/dec_cond_vae.yaml')
parser.add_argument('--wandb_name',
                    type=str,
                    default="pcvae_test_run",
                    help="Name of the wandb run")
parser.add_argument('--wandb_toggle', '-w',
                    type=bool,
                    default=False,
                    help="Toggle to enable or disable Wandb logging")
parser.add_argument('--wandb_project',
                    type=str,
                    default="pytorch-vae-run",
                    help="Name of the registered Wandb project")
parser.add_argument('--wandb_entity',
                    type=str,
                    default="carff",
                    help="Name of the registered Wandb entity")
parser.add_argument('--kld_scheduler',
                    type=bool,
                    default=False,
                    help="Turn on delayed linear KLD scheduling based on parameters in config file")
parser.add_argument('--load_ckpt',
                    type=str,
                    default="",
                    help="Load the checkpoint using the path")

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


if args.wandb_toggle:
    tb_logger = WandbLogger(
        project=args.wandb_project,
        save_dir=config['logging_params']['save_dir'],
        name=config['model_params']['name'],)
    tb_logger.log_dir = tb_logger.save_dir
else:
    tb_logger =  NormalLogger(save_dir=config['logging_params']['save_dir'],
                                   name=config['model_params']['name'],)

seed_everything(config['exp_params']['manual_seed'], True)

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

data.setup()

try:
    num_dataset_views = data.train_dataset.num_views + 1
    print("NUM VIEWS TRAIN", num_dataset_views)
    print("NUM VIEWS VALID", data.val_dataset.num_views + 1)
    print("NUM SCENES", data.val_dataset.num_scenes + 1)

    num_dataset_scenes = data.train_dataset.num_scenes + 1

    model = vae_models[config['model_params']['name']](**config['model_params'], 
                                                        num_decoder_classes=num_dataset_views, 
                                                        num_scenes=num_dataset_scenes)
    
except KeyError:
    print(f"Couldn't find VAE model {config['model_params']['name']}")
    print("Available models:\n", "\n".join([s for s in vae_models.keys()]))
    raise

wandb_args = {
    "use_wandb": args.wandb_toggle,
}

# setup wandb
if args.wandb_toggle:
    wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config, name=args.wandb_name)
    wandb_args["wandb_project"] = args.wandb_project
    wandb_args["wandb_run"] = args.wandb_run

if args.load_ckpt == "":
    experiment = VAEXperiment(model,
                        config['exp_params'],
                        **wandb_args,
                        kld_scheduler=args.kld_scheduler,
                        log_params=config['logging_params'])
else:
    experiment = VAEXperiment.load_from_checkpoint(args.load_ckpt,
                        vae_model=model,
                        params=config['exp_params'],
                        **wandb_args,
                        kld_scheduler=args.kld_scheduler,
                        log_params=config['logging_params'])

runner = Trainer(logger=tb_logger,
                callbacks=[
                    LearningRateMonitor(),
                    ModelCheckpoint(every_n_epochs=10,
                                    save_top_k=2,
                                    dirpath=os.path.join(tb_logger.log_dir , "checkpoints"),
                                    monitor="val_loss",
                                    save_last= True),
                ],
                strategy=DDPPlugin(find_unused_parameters=False),
                **config['trainer_params'])


Path(f"{tb_logger.save_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.save_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.save_dir}/embeddings").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
