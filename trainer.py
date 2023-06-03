import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from config import get_config
from vqgan import VQGAN
from data.custom import CustomTrain, CustomTest



class CustomDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size,
                 num_workers,
                 training_images_list_file,
                 test_images_list_file)
                 )
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.training_images_list_file = training_images_list_file
        self.test_images_list_file = test_iamges_list_file

    def setup(self, stage=None):
        self.train_dataset = CustomTrain(self.training_images_list_file, size = 256)
        self.val_dataset =  CustomTest(self.test_images_list_file, size = 256)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)




class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TensorBoardLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(tag, grid,global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")



if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    opt = get_config()
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    model = VQGAN(opt.model.ch,  #[64,128,256]
                  opt.model.num_res_blocks, #2
                  opt.model.attn_resolutions,  #
                  opt.model.resamp_with_conv,
                  opt.model.in_channels, #3
                  opt.model.resolution, 
                  opt.model.z_channels,
                  opt.model.num_codebook_vectors,
                  opt.model.latent_dim,
                  opt.model.ckpt_path=None,
                  ignore_keys=[],
                  image_key="image",
                  colorize_nlabels=None,
                  monitor=None,
                  warmup_steps = opt.lr.warmup_steps,
                  min_lr = opt.lr.min_lr,
                  max_lr = opt.lr.max_lr,
                  max_decay_steps = opt.lr.max_decay_steps,
                  disc_start = opt.weight.disc_start, 
                  codebook_weight= opt.weight.codebook_weight,  
                  pixelloss_weight= opt.weight.pixelloss_weight,
                  disc_n_layers= opt.model.disc_n_layers,  
                  num_filters_last = opt.model.num_filters_last,
                  disc_factor= opt.weight.disc_factor,         #1.0    
                  disc_weight= opt.weight.disc_weight,         #0.1
                  perceptual_weight= opt.weight.perceptual_weight,   #0.1
                  disc_loss= opt.model.disc_loss)
    
    #build training loggers 
    trainer_loggers = []
    trainer_loggers.append(pytorch_lightning.loggers.WandbLogger(name = nowname,
                                                                 save_dir = logdir, 
                                                                 offline = False,
                                                                 id = nowname))
    trainer_loggers.append(pytorch_lightning.loggers.TensorBoardLogger(name = nowname,
                                                                       save_dir = logdir))
    trainer_callbacks = []
    trainer_callbacks.append(pytorch_lightning.callbacks.ModelCheckpoint(dirpath = ckptdir,
                                                                         filename = "{epoch:06}",
                                                                         verbose = True,
                                                                         save_last = True,
                                                                         monitor = model.monitor,
                                                                         save_top_k = 5))
    trainer_callbacks.append(ImageLogger(batch_frequency = 750,
                                         max_images = 4,
                                         clamp = True))
    trainer_callbacks.append(LearningRateMonitor("logging_interval": "step"))

    trainer = Trainer(accelerator = opt.accelerator,
                    devices= opt.devices,
                    callbacks = trainer_callbacks,
                    logger = trainer_loggers)

    # data
    data = CustomDataModule(batch_size= opt.data.batch_size,
                            num_workers = opt.data.num_workers,
                            training_images_list_file = opt.data.training_images_list_file,
                            test_images_list_file = opt.data.test_images_list_file)

    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb; pudb.set_trace()

    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # run
    if opt.resume:
        trainer.load_from_checkpoint(opt.checkpoint)
    try:
        trainer.fit(model, data)
        except Exception:
             melk()
             raise