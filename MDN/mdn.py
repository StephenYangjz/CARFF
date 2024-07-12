import os
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Mixture Density Network model as described in the CARFF paper.
"""
class MDN(nn.Module):
   
    def __init__(self, dim_in, dim_out, n_components, hidden_dim,):
        super().__init__()
        num_sigma_channels =  dim_out * n_components
        self.dim_in, self.dim_out, self.n_components = dim_in, dim_out, n_components
        
        self.base = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_out * n_components)
            )
        
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_sigma_channels)
            )

        self.pi_head = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_components),
        )

    def forward(self, x, eps=1e-6):
        shared_output = F.relu(self.base(x))
        pis = self.pi_head(x)
        log_pi = torch.log_softmax(pis, dim=-1)
        mu = self.mu_head(shared_output)
        sigma = self.sigma_head(shared_output)

        sigma = torch.exp(sigma + eps)
        mu = mu.reshape(-1, self.n_components, self.dim_out)
        sigma = sigma.reshape(-1, self.n_components, self.dim_out)
        return log_pi, mu, sigma

    def loss_mod(self, mu, sigma, log_pi, y):
        z_score = (y - mu) / sigma
        
        if len(y.shape) == 3:
            normal_loglik = (
                -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
                -torch.sum(torch.log(sigma), dim=-1)
            )
        else:
            normal_loglik = (
                -0.5 * torch.einsum("bijc,bijc->bij", z_score, z_score)
                - torch.sum(torch.log(sigma), dim=-1)
            )


        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik.sum()

    """
        Samples the weights, mu and sigma for the next scene from the MDN provided the previous scene latent.
    """
    def sample(self, x):
        log_pi, mu, sigma = self.forward(x)
        weights = torch.exp(log_pi)
        cum_pi = torch.cumsum(weights, dim=-1)
        rvs = torch.rand(len(x), 1).to(x)
        rand_pi = torch.searchsorted(cum_pi, rvs)
        rand_normal = torch.randn_like(mu) * sigma + mu
        samples = torch.take_along_dim(rand_normal, indices=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)
        return samples, weights, mu, sigma

    def save(self, path):
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        
        torch.save(checkpoint, os.path.join(path, "checkpoint.pth"))

    def load(self, path):
        checkpoint = torch.load(os.path.join(path, "checkpoint.pth"))
        self.load_state_dict(checkpoint["model_state_dict"])
                
        return checkpoint