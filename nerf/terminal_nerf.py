import sys
import torch
import argparse
from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *
from functools import partial
from loss import huber_loss
from init import init
from nerf.network_tcnn import NeRFNetwork

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and process dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="File path for the dataset.")
    args = parser.parse_args()
    
    opt = init(args.dataset_path)
    seed_everything(0)
    assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )

    criterion = torch.nn.MSELoss(reduction='none')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
    train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
    trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=[PSNRMeter()], use_checkpoint=opt.ckpt, eval_interval=50)
    gui = NeRFGUI(opt, trainer, train_loader)

    gui.render_no_gui(1800)

