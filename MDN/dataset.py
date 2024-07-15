import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

class TensorDataset(Dataset):
    def __init__(self, all_mus, all_vars, all_latents, perturb_range, device):
        assert len(all_mus) == len(all_vars)
        self.device = device
        self.all_mus = self.lists_to_tensors(all_mus)
        self.all_vars = self.lists_to_tensors(all_vars)
        self.all_latents = self.lists_to_tensors(all_latents)
        self.num_scenes = len(all_mus)
        self.num_views = len(all_mus[list(all_mus.keys())[0]])
        self.num_scenes_temp = self.num_scenes + 1
        self.perturb_range = perturb_range
    
    def lists_to_tensors(self, dictionary):
        for key, value in dictionary.items():
            dictionary[key] = torch.tensor(value).to(self.device)
        return dictionary

    def __len__(self):
        return len(self.all_mus[0])

    def __getitem__(self, idx):
        factor = torch.randint(0,2, (1,)).item()
        scene = torch.randint(0, self.num_scenes_temp // 2 - 1, (1,)).item()

        start_scene = factor*(self.num_scenes_temp // 2) + (scene)  
        # one_hot_encode = F.one_hot(torch.tensor([scene]), self.num_scenes_temp -1).to(self.device)

        end_scene = start_scene + 1
    
        start_mu = self.all_mus[start_scene][idx]
        # start_latents = self.all_latents[start_scene][idx]
        start_var = self.all_vars[start_scene][idx]
        
        idx = idx + torch.randint(-self.perturb_range[0], self.perturb_range[1], (1,))[0]
        if idx >= self.num_views:
            idx = self.num_views - 1
        end_mu = self.all_mus[end_scene][idx]
        end_var = self.all_vars[end_scene][idx]
        # end_latents = self.all_latents[end_scene][idx]

        gaussian_noise1 = torch.randn(start_mu.shape).cuda() * 0.005
        gaussian_noise2 = torch.randn(start_var.shape).cuda() * 0.005

        return torch.cat([start_mu + gaussian_noise1, start_var + gaussian_noise2]), torch.cat([end_mu, end_var])
