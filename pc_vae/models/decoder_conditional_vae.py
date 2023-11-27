import torch
from models import VanillaVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from vit_pytorch import SimpleViT
from torchvision.models import vit_l_16, resnet50, ResNet50_Weights, mobilenet_v2, MobileNet_V2_Weights

class DecoderConditionalVAE(VanillaVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 decoder_inputs_onehot: bool,
                 num_decoder_classes: int,
                 num_scenes: int,
                 hidden_dims: List = None,
                 enable_cheating: bool = False,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.enable_cheating = enable_cheating
        self.latent_dim = latent_dim
        self.num_decoder_classes = num_decoder_classes
        self.decoder_inputs_onehot = decoder_inputs_onehot
        self.num_scenes = num_scenes

        if decoder_inputs_onehot:
            self.extra_decoder_inputs = num_decoder_classes
        else:
            self.extra_decoder_inputs = 1

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels=3,
                kernel_size= 3, stride= 1, padding  = 1),
             nn.BatchNorm2d(3),
             nn.LeakyReLU()))

        vit = vit_l_16(
            weights="IMAGENET1K_V1",
            image_size=224
        )
        modules.append(vit)

        self.fc_mu = nn.Linear(1000, latent_dim)
        self.fc_var = nn.Linear(1000, latent_dim)

        self.encoder = nn.Sequential(*modules)
        
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                param.requires_grad = False

        for param in self.encoder[0].parameters():
            param.requires_grad = True

        modules = []

        if self.enable_cheating:
            self.extra_decoder_inputs += self.num_scenes
        self.decoder_input = nn.Linear(latent_dim + self.extra_decoder_inputs, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=4,
                                               padding=1,
                                               output_padding=3),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def decode(self, z: Tensor, v: Tensor, debug_info: dict = None) -> Tensor:
        if self.enable_cheating:
            assert debug_info is not None
        assert z.shape[0] == v.shape[0], (z.shape, v.shape)

        if self.decoder_inputs_onehot:
            v = F.one_hot(v, num_classes=self.num_decoder_classes)
            assert v.shape == (z.shape[0], self.num_decoder_classes), (z.shape, v.shape)
        else:
            if len(v.shape) == 1:
                v = v.unsqueeze(-1)
            assert v.shape == (z.shape[0], 1), (z.shape, v.shape)
        z_and_v = torch.cat([z, v], dim = 1).cuda()
        if self.enable_cheating:
            scene_id = debug_info
            scene_id = F.one_hot(scene_id, num_classes=self.num_scenes).cuda()
            z_and_v = torch.cat([z_and_v, scene_id], dim = 1)

        result = self.decoder_input(z_and_v)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, img1: Tensor, pose2: Tensor, debug_info: dict = None) -> List[Tensor]:
        mu, log_var = self.encode(img1)
        z = self.reparameterize(mu, log_var)

        decoded = self.decode(z, pose2)
        return  [decoded, input, mu, log_var]

    def forward_mean(self, input: Tensor, labels: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = mu.detach()
        view = labels
        return  [self.decode(z, view), input, mu, log_var]

    def sample_views(self, num_samples: int):
        v = torch.randint(low=0, high=self.num_decoder_classes, size=(num_samples,))
        return v

    def sample(self,
               num_samples:int,
               current_device: int,
               views: Tensor = None,
               **kwargs) -> Tensor:
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)

        if views is None:
            views = self.sample_views(num_samples)
            views = views.to(current_device)

        samples = self.decode(z, views)
        return samples

    def generate(self, x: Tensor, v: Tensor = None, debug_info: dict = None, **kwargs) -> Tensor:
        if v is None:
            v = self.sample_views(x.shape[0])
            v = v.to(x.device)
        
        assert x.shape[0] == v.shape[0], (x.shape, v.shape)
        return self.forward(x, v, debug_info=debug_info)[0]

    def generate_mean(self, x: Tensor, v: Tensor = None, **kwargs) -> Tensor:
        if v is None:
            v = self.sample_views(x.shape[0])
            v = v.to(x.device)

        assert x.shape[0] == v.shape[0], (x.shape, v.shape)
        return self.forward_mean(x, v)[0]
