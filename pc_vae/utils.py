import pytorch_lightning as pl
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio
from pytorch_lightning.loggers import TensorBoardLogger

PSNR = PeakSignalNoiseRatio().cuda()

def data_loader(fn):
    def func_wrapper(self):
        try:
            return pl.data_loader(fn)(self)

        except:
            return fn(self)

    return func_wrapper

def save_image_with_axes(title, x_label, y_label, *args, **kwargs):
    vutils.save_image(*args, **kwargs)

class NormalLogger(TensorBoardLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def log_dir(self):
        return self.save_dir
