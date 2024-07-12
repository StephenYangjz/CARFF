import os
import math
import torch
from torch import optim
from torch.nn import functional as F
from models.types_ import *
from models import BaseVAE, DecoderConditionalVAE
from utils import data_loader, save_image_with_axes, PSNR
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
import numpy as np
import cv2 as cv
from torchvision.datasets.folder import default_loader
import json
from scipy.stats import norm

"""
    VAEXperiment class sets up the pipeline for training PC-VAE using PyTorch Lightning modules. W&B logging, KLD scheduling
    and other parameters from the config files and command line flags in the README can be specified to modify the training. 
"""
class VAEXperiment(pl.LightningModule):

    def __init__(self,
                vae_model: BaseVAE,
                params: dict,
                use_wandb: bool = False, 
                wandb_project: str = None,
                wandb_run: any = None,
                kld_scheduler:bool = False,
                debug:bool = True,
                log_params: dict = {}) -> None:
        super(VAEXperiment, self).__init__()
        self.log_params = log_params

        self.train_psnrs = []
        self.val_psnrs = []

        self.kld_weights = [params['kld_weight']]
        self.kld_scheduler = kld_scheduler

        # KLD scheduler described in the CARFF paper used to train the optimal version of PC-VAE
        def kld_exp_func(x):
            epoch_start_to_change = params['kld_start']
            converged_epoch = params['kld_end']
            
            if x < epoch_start_to_change:
                return params['kld_weight']
            
            elif x < converged_epoch:
                increment = (params['kld_max'] - params['kld_weight']) / (converged_epoch - epoch_start_to_change)
                return params['kld_weight'] + increment * (x-epoch_start_to_change)
            else:
                return params['kld_max']
        self.kld_exp_func = kld_exp_func
        
        self.model = vae_model
        self.conditional_decoder = isinstance(self.model, DecoderConditionalVAE)
        self.params = params
        self.curr_device = None
        self.use_wandb = use_wandb
        if self.use_wandb:
            self.wandb_log = {}
            self.wandb_run = wandb_run
            self.wandb_project = wandb_project

    def forward(self, img1: Tensor, img2_pose: Tensor, **kwargs) -> Tensor:
        return self.model(img1, img2_pose, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        img1_pose, img1, img2_pose, img2, _ = batch
        self.curr_device = img1.device

        img2_hat, _, mu, log_var = self.forward(img1, img2_pose)
        results = [img2_hat, img2, mu, log_var]

        train_loss = self.model.loss_function(*results,
                                                M_N = self.params['kld_weight'],
                                                optimizer_idx=optimizer_idx,
                                                batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']


    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        img1_pose, img1, img2_pose, img2, _ = batch
        self.curr_device = img1.device

        img2_hat, _, mu, log_var = self.forward(img1, img2_pose)
        results = [img2_hat, img2, mu, log_var]

        val_loss = self.model.loss_function(*results,
                                        M_N = 1.0,
                                        optimizer_idx = optimizer_idx,
                                        batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
    
    def generate_and_save_embeddings(self, transforms_source, transforms_dest):
        dataset = self.trainer.datamodule.train_dataset
        dataloader = self.trainer.datamodule.train_dataloader()
        batch_size = 64

        json_file = os.path.join(dataset.data_dir, transforms_source)
        print(f"Loading file {json_file}")
        with open(json_file, 'r') as fh:
            json_data = json.load(fh) 

        image_paths = []

        for frames in json_data['frames']:
            image_path = frames['file_path']
            image_path = os.path.join(dataset.data_dir, image_path)
            image_paths.append(image_path)

        latents = []
        mus = []
        log_vars = []
        for i in range(0, len(image_paths) // batch_size):
            image_paths_batch = image_paths[i*batch_size:(i+1)*batch_size]
            image_batch = [dataset.transforms(default_loader(path)).unsqueeze(0) for path in image_paths_batch]
            image_batch = torch.vstack(image_batch)
            mu, log_var = self.model.encode(image_batch.cuda())
            z = self.model.reparameterize(mu, log_var)
            mus.extend(mu)
            log_vars.extend(log_var)
            latents.extend(z)
        image_paths_batch = image_paths[-(len(image_paths) % batch_size):]
        image_batch = [dataset.transforms(default_loader(path)).unsqueeze(0) for path in image_paths_batch]
        image_batch = torch.vstack(image_batch)
        mu, log_var = self.model.encode(image_batch.cuda())
        z = self.model.reparameterize(mu, log_var)
        mus.extend(mu)
        log_vars.extend(log_var)
        latents.extend(z)

        new_json_file = os.path.join(self.log_params['save_dir'], 'embeddings', transforms_dest)

        with open(json_file, 'r') as f:
            transforms = json.load(f)
            for index, frame in enumerate(transforms["frames"]):
                frame["latents"] = latents[index].tolist()
                frame["mu"] = mus[index].tolist()
                frame["var"] = log_vars[index].tolist()

        with open(new_json_file, 'w') as f:
            json.dump(transforms, f, indent=4)

        print(f'Saved embeddings to {transforms_dest}')
        if self.use_wandb:
            artifact = wandb.Artifact(name=transforms_dest, type="file")
            artifact.add_file(local_path=new_json_file)
            self.wandb_run.log_artifact(artifact)
    
    def generate_and_save_val_embeddings(self):
        self.generate_and_save_embeddings('transforms_val.json', f'transforms_val{self.current_epoch}.json')
    
    def generate_and_save_train_embeddings(self):
        self.generate_and_save_embeddings('transforms_train.json', f'transforms_train{self.current_epoch}.json')

    """
        Utilizes the KLD scheduler function kld_exp_func to modify the KLD weight every epoch.
    """
    def increment_kld(self):
        if self.params['kld_weight'] < self.params['kld_max']:
            self.params['kld_weight'] = self.kld_exp_func(self.current_epoch + 1)
        self.kld_weights.append(self.params['kld_weight'])
        
        data = [[x, y] for (x, y) in zip(np.arange(0, self.current_epoch + 1), self.kld_weights)]
        if self.use_wandb:
            kld_plot = wandb.Table(data=data, columns=["Epochs", "KLD Weights"])
            self.wandb_log['KLD Weight Increments'] = wandb.plot.line(kld_plot, "Epochs", "KLD Weights", title="KLD Weights vs Epochs")

    """
        Logs the PSNR of the predictions on the training data.
    """
    def log_train_psnr(self):
        train_metrics_loader = self.trainer.datamodule.train_metrics_dataloader()
        train_metrics_iter = iter(train_metrics_loader)
        psnrs = []
        for img1_pose, img1, img2_pose, img2, _ in train_metrics_iter:
            img2 = F.interpolate(img2, [128, 128], mode='bilinear', align_corners=True)
            img2_hat, _, _, _ = self.forward(img1.cuda(), img2_pose.cuda())
            for i in range(len(img2)):
                psnrs.append(PSNR(img2_hat[i].cuda(), img2[i].cuda()).item())
        avg_psnr = np.mean(psnrs)
        print("Average Train PSNR:", avg_psnr)
        if self.use_wandb:
            self.wandb_log['Average Train PSNR'] = avg_psnr
        
    """
        Logs the PSNR of the predictions on the validation data.
    """
    def log_val_psnr(self):
        viz_indices = [6, 34, 62]
        val_dataset = self.trainer.datamodule.val_dataset
        train_metrics_dataset = self.trainer.datamodule.train_metrics_dataset
        val_metrics_loader = self.trainer.datamodule.val_dataloader()
        val_metrics_iter = iter(val_metrics_loader)
        psnrs = []
        for img1_pose, img1, img2_pose, _, info in val_metrics_iter:
            img1_scene = info['scene_id']
            img2_hat, _, _, _ = self.forward(img1.cuda(), img2_pose.cuda())
            for i, scene_id in enumerate(img1_scene):
                idx = val_dataset.scene_and_view_to_idx[(scene_id.item(), img2_pose[i].item())]
                img2 = train_metrics_dataset.transforms(default_loader(train_metrics_dataset.imgs[idx]['image_path']))
                img2 = F.interpolate(img2.unsqueeze(0), [128, 128], mode='bilinear', align_corners=True)
                img1 = F.interpolate(img1, [128, 128], mode='bilinear', align_corners=True)

                if i in viz_indices:
                    logged_img = torch.cat([img1[i].cuda(), img2.squeeze(0).cuda(), img2_hat[i].cuda()], dim=1)
                    if self.use_wandb:
                        wandb_img = wandb.Image(
                            logged_img,
                            caption=f'Input Image {i}, Image 2 GT, Image 2 Hat'
                        )
                        self.wandb_log[f'Image {i} Input->Recons'] = wandb_img

                psnrs.append(PSNR(img2_hat[i].cuda(), img2.cuda()).item())
        avg_psnr = np.mean(psnrs)
        print("Average Validation PSNR:", avg_psnr)
        if self.use_wandb:
            self.wandb_log['Average Validation PSNR'] = avg_psnr

    def on_validation_end(self) -> None:
        self.sample_images()

        if self.kld_scheduler:
            self.increment_kld()

        if self.current_epoch % 20 == 0:
            self.generate_and_save_train_embeddings()
            self.generate_and_save_val_embeddings()

        self.log_train_psnr()
        self.log_val_psnr()

        if self.use_wandb:
            wandb.log(self.wandb_log)
            self.wandb_log = {}

    """
        Samples images from different poses to generate model reconstructions as a grid as displayed in the CARFF paper. 
    """
    def sample_images(self):
        loader = self.trainer.datamodule.train_dataloader()
        dataset = self.trainer.datamodule.train_dataset
        img1_pose, img1, _, _, info = next(iter(loader))

        N_VAL_IMAGES = 8

        assert N_VAL_IMAGES <= img1_pose.shape[0], (N_VAL_IMAGES, img1_pose.shape)
        test_img = img1[:N_VAL_IMAGES,...]
        test_img = test_img.to(self.curr_device)

        N_VAL_VIEWS = 8 

        new_poses = torch.randint(low=0, high=dataset.num_views + 1, size=(N_VAL_VIEWS,))
        new_poses = new_poses.to(self.curr_device)

        N_VAL_TOTAL = N_VAL_IMAGES*N_VAL_VIEWS
        test_input = torch.repeat_interleave(test_img,N_VAL_VIEWS,dim=0)

        test_label = torch.tile(new_poses,(N_VAL_IMAGES,))

        assert test_label.shape  == (N_VAL_TOTAL,), test_label.shape

        scene_id = info['scene_id'][:N_VAL_IMAGES,...].to(self.curr_device)
        scene_id = torch.repeat_interleave(scene_id,N_VAL_VIEWS,dim=0)
        recons = self.model.generate(test_input, v = test_label)

        img_title = f"Encoder inputs (epoch {self.current_epoch})"
        img_name  = f"recons_{self.logger.name}_Epoch_{self.current_epoch}_input.png"
        img_path  = os.path.join(self.logger.log_dir, "Reconstructions", img_name)

        save_image_with_axes(
            img_title,
            "Random camera angles of the same configuration" if self.conditional_decoder else None,
            "Random scene configurations" if self.conditional_decoder else None,
            test_img.data,
            img_path,
            normalize=True,
            nrow=N_VAL_VIEWS if self.conditional_decoder else N_VAL_TOTAL)

        if self.use_wandb:
            self.wandb_log['Encoder inputs'] = wandb.Image(img_path, caption=img_name)

        img_title = f"Encoder->Decoder, sampling z (epoch {self.current_epoch})"
        img_name = f"enc_dec_{self.logger.name}_Epoch_{self.current_epoch}_output.png"
        img_path = os.path.join(self.logger.log_dir, "Reconstructions", img_name)

        enc_and_recons = recons.reshape(N_VAL_IMAGES,N_VAL_VIEWS,3,128,128)
        test_img = F.interpolate(test_img, [128, 128], mode='bilinear', align_corners=True)
        enc_og_view = test_img.unsqueeze(1)

        enc_and_recons = cv.normalize(enc_and_recons.cpu().numpy(),  None, 0, 255, cv.NORM_MINMAX)
        enc_og_view = cv.normalize(enc_og_view.cpu().numpy(),  None, 0, 255, cv.NORM_MINMAX)

        enc_and_recons = torch.Tensor(enc_and_recons).cuda()
        enc_og_view = torch.Tensor(enc_og_view).cuda()
        separator = torch.zeros_like(enc_og_view).cuda()

        dataset = self.trainer.datamodule.train_dataset
        transform = transforms.Compose([
            transforms.Resize(self.trainer.datamodule.patch_size),             
        ])
        enc_og_view = torch.cat([enc_og_view, separator], dim = 1).cuda()

        GT_views = img1_pose[:N_VAL_IMAGES,...]
        for scene_id in range(dataset.num_scenes + 1):
            col = []
            for view_id in GT_views:
                idx = dataset.scene_and_view_to_idx[(scene_id, int(view_id.item()))]
                retrieved_img = dataset.imgs[idx]['image_path']
                col.append(np.array(transform(default_loader(retrieved_img))))

            col = torch.Tensor(np.array(col)).cuda()
            col = col[:, None, :, :, :].movedim(4, 2)

            enc_og_view = torch.cat([enc_og_view, col], dim = 1)
        enc_og_view = torch.cat([enc_og_view, separator], dim=1)
        enc_and_recons = torch.cat([enc_og_view, enc_and_recons], dim=1)

        assert enc_and_recons.shape == (N_VAL_IMAGES, N_VAL_VIEWS + dataset.num_scenes + 4,3,128,128)
        enc_and_recons = enc_and_recons.reshape((N_VAL_IMAGES*(N_VAL_VIEWS + dataset.num_scenes + 4),3,128,128))

        save_image_with_axes(
            img_title,
            "View",
            "Scene (random view)",
            enc_and_recons.data,
            img_path,
            normalize=True,
            nrow=(N_VAL_VIEWS+ dataset.num_scenes + 4),

            xticks=(np.arange(0, N_VAL_VIEWS) * 128 + 50 + 128 * (dataset.num_scenes + 4), [str(i) for i in new_poses.cpu().numpy().tolist()]),
        )
        if self.use_wandb:
            self.wandb_log['Enc. input + dec. outputs (sampling z)'] = wandb.Image(img_path, caption=img_name)

        img_title = f"Decoder outputs, sampling z (epoch {self.current_epoch})"
        img_name = f"recons_{self.logger.name}_Epoch_{self.current_epoch}_output.png"
        img_path = os.path.join(self.logger.log_dir, "Reconstructions", img_name)

        save_image_with_axes(
            img_title,
            "Random camera angles of the same configuration" if self.conditional_decoder else None,
            "Random scene configurations" if self.conditional_decoder else None,
            recons.data,
            img_path,
            normalize=True,
            nrow=N_VAL_VIEWS if self.conditional_decoder else N_VAL_TOTAL)
        if self.use_wandb:
            self.wandb_log['Decoder outputs (sampling z)'] = wandb.Image(img_path, caption=img_name)

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
