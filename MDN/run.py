import os
import yaml
import torch
import json
import tqdm
import argparse
from torch.quasirandom import SobolEngine
from mdn import MDN
from dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

"""
    Extracts the latents, mus and vars from the transforms data
"""
def extract_info(data):
    latents = {}
    mus = {}
    log_vars = {}
    for frame in data["frames"]:
        # latents
        if frame["scene_id"] in latents.keys():
            latents[frame["scene_id"]].append(frame["latents"])
        else:
            latents[frame["scene_id"]] = [frame["latents"]]
        # mu
        if frame["scene_id"] in mus.keys():
            mus[frame["scene_id"]].append(frame["mu"])
        else:
            mus[frame["scene_id"]] = [frame["mu"]]
        # var
        if frame["scene_id"] in log_vars.keys():
            log_vars[frame["scene_id"]].append(frame["var"])
        else:
            log_vars[frame["scene_id"]] = [frame["var"]]
    return latents, mus, log_vars

"""
    Main function to run the MDN training.
"""
def main(transform_path, save_model_path, config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    model_params, exp_params = config["model_params"], config["exp_params"]

    f = open(transform_path, "r")
    data = json.load(f)

    latents, mus, log_vars = extract_info(data)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sobol_engine = SobolEngine(dimension=model_params["latent_dim"], scramble=True)
    model = MDN(2 * model_params["latent_dim"],model_params["latent_dim"], model_params["K"], model_params["hidden_dim"]).to(device)

    dataset = TensorDataset(mus, log_vars, latents)
    dataloader = DataLoader(dataset, batch_size=exp_params["batch_size"], shuffle=True)

    progress_bar = tqdm.tqdm(range(exp_params["num_epochs"]), desc="Loss: ---", position=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=exp_params["lr"])
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=10000)

    losses = []

    for epoch in progress_bar:
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            weights, mu_preds, var_preds = model(batch_X)

            samples_sobol = sobol_engine.draw(exp_params["num_samples"]).to(torch.float).to(device)
            samples_sobol = samples_sobol[None,:,None,:].repeat(batch_y.shape[0], 1, model_params[K], 1)
            
            batch_y = batch_y[:,None,None,:].repeat(1, exp_params["num_samples"],model_params["K"],1)
            latent_vectors = samples_sobol * torch.exp(batch_y[..., model_params["latent_dim"]:]) + batch_y[...,:model_params["latent_dim"]]
            loss = model.loss_mod(mu_preds[:,None,...].repeat(1, exp_params["num_samples"], 1, 1),var_preds[:,None,...].repeat(1, exp_params["num_samples"], 1, 1),weights[:,None].repeat(1, exp_params["num_samples"], 1),latent_vectors)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        losses.append(loss.item())
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(losses[50:])), losses[50:])
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("mdn_loss.png")
    plt.close()

    model.save(os.path.dirname(save_model_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with specified configurations.")
    parser.add_argument("--transform_path", type=str, required=True, help="File path for the transforms.")
    parser.add_argument("--save_model_path", type=str, required=True, help="File path to save the model.")
    parser.add_argument("--config_path", type=str, default="./mdn_config.yaml", help="File path for the MDN training and model configs.")
    
    args = parser.parse_args()
    main(args.transform_path, args.save_model_path)

