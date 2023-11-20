from .base import *
from .vanilla_vae import *
from .decoder_conditional_vae import *


# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE

vae_models = {'VanillaVAE':VanillaVAE,
              'DecoderConditionalVAE':DecoderConditionalVAE}
