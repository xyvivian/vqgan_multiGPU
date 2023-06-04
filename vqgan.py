import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from models.model import Encoder, Decoder
from models.codebook import Codebook
from models.losses import VQLoss
from lr_scheduler import LambdaWarmUpCosineScheduler


class VQGAN(pl.LightningModule):
    def __init__(self,
                 ch,  #[64,128,256]
                 num_res_blocks, #2
                 attn_resolutions,  #
                 resamp_with_conv, #True
                 in_channels, #3
                 resolution, 
                 z_channels,
                 num_codebook_vectors,
                 latent_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 warmup_steps = 1000,
                 min_lr = 0.01,
                 max_lr = 0.1,
                 max_decay_steps = 500,
                 disc_start = 2000, 
                 codebook_weight=1.0,  
                 pixelloss_weight=1.0,
                 disc_n_layers=3,  
                 num_filters_last = 64,
                 disc_factor=1.0,         #1.0    
                 disc_weight=1.0,         #0.1
                 perceptual_weight=1.0,   #0.1
                 disc_loss="hinge",
                 beta1 = 0.5,
                 beta2 = 0.99,
                 ):
        super().__init__()
        self.image_key = image_key
        self.monitor = monitor
        self.encoder = Encoder(ch = ch,  #[64,128,256]
                               num_res_blocks = num_res_blocks, #2
                               attn_resolutions = attn_resolutions,  #
                               resamp_with_conv= resamp_with_conv,
                               in_channels = in_channels, #3
                               resolution = resolution, 
                               z_channels = z_channels)
        self.decoder = Decoder(ch = ch, 
                               out_ch = in_channels, 
                               num_res_blocks = num_res_blocks,
                               attn_resolutions = attn_resolutions,
                               resamp_with_conv= resamp_with_conv, 
                               resolution = resolution, 
                               z_channels = z_channels, 
                              give_pre_end=False)

        self.loss = VQLoss(disc_start= disc_start, 
                            codebook_weight = codebook_weight,  
                            pixelloss_weight=pixelloss_weight,
                            n_layers=disc_n_layers, 
                            in_channels= in_channels, 
                            num_filters_last = num_filters_last,
                            disc_factor= disc_factor,         #1.0    
                            disc_weight= disc_weight,         #0.1
                            perceptual_weight=perceptual_weight,   #0.1
                            disc_loss=disc_loss)

        self.codebook = Codebook(num_codebook_vectors = num_codebook_vectors, 
                                 latent_dim = latent_dim,
                                 beta = 0.25)

        self.quant_conv = torch.nn.Conv2d(z_channels, latent_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(latent_dim, z_channels, 1)
        self.beta1 = beta1
        self.beta2 = beta2
        self.warmup_steps = warmup_steps
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_decay_steps = max_decay_steps

    ## TODO: add the lr_scheduler to the learning process
    ## VERY IMPORTANT!!!
    def configure_optimizers(self):
        lr = self.min_lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.codebook.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(self.beta1, self.beta2))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(self.beta1, self.beta2))
        ae_scheduler = LambdaWarmUpCosineScheduler(opt_ae, 
                                                   lr_min= self.min_lr,
                                                   lr_max= self.max_lr, 
                                                   lr_start=self.min_lr,
                                                   max_decay_steps=self.max_decay_steps,
                                                   warmup_steps = self.warmup_steps)
        disc_scheduler = LambdaWarmUpCosineScheduler(opt_disc, 
                                                   lr_min= self.min_lr,
                                                   lr_max= self.max_lr, 
                                                   lr_start=self.min_lr,
                                                   max_decay_steps=self.max_decay_steps,
                                                   warmup_steps = self.warmup_steps)
        return [opt_ae, opt_disc], [ae_scheduler,disc_scheduler]
    

    def init_from_ckpt(self, 
                       path,
                       ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Successfully Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, min_encoding_indices = self.codebook(h)
        return quant, emb_loss, min_encoding_indices

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()


    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            ae_scheduler, _ = self.lr_schedulers()
            ae_scheduler.step(self.global_step)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            _, disc_scheduler = self.lr_schedulers()
            disc_scheduler.step(self.global_step)
            return discloss


    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        d_factor = log_dict_ae["val/d_weight"]
        self.log("val/rec_loss", rec_loss.item(),
                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss.item(),
                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/d_factor", d_factor.item(),
                prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/discloss", discloss.item(),
                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", log_dict_ae["val/total_loss"].item(),prog_bar= False, 
                                logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/quant_loss", log_dict_ae["val/quant_loss"].item(),prog_bar= False, 
                                logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/nll_loss", log_dict_ae["val/nll_loss"].item(),prog_bar= False, 
                                logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/g_loss", log_dict_ae["val/g_loss"].item(),prog_bar= False, 
                                logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return {"rec_loss": rec_loss, "ae_loss": aeloss}
    


    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
