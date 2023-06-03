import torch.nn as nn
from utils import ResidualBlock, AttnBlock, UpSample, GroupNorm, Swish,DownSample

class Encoder(nn.Module):
    def __init__(self, 
                ch,  #[64,128,256]
                num_res_blocks, #2
                attn_resolutions,  #
                resamp_with_conv=True,
                in_channels, #3
                resolution, 
                z_channels, #[512]
                **ignorekwargs):
        super(Encoder, self).__init__()
        channels = ch
        latent_dim = z_channels
        
        #downsampling
        layers = [nn.Conv2d(in_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i] #block_in
            out_channels = channels[i + 1] #block_out
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(AttnBlock(in_channels))
            if i != len(channels)-2:
                layers.append(DownSample(channels[i+1], resampe_with_conv))
                resolution //= 2
        
        #middle 
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(AttnBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
       
        #end 
        layers.append(nn.Conv2d(channels[-1],latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class Decoder(nn.Module):
    def __init__(self, 
                 ch, 
                 out_ch, 
                 num_res_blocks,
                 attn_resolutions,
                 resamp_with_conv=True, 
                 resolution, 
                 z_channels, 
                 give_pre_end=False, 
                 **ignorekwargs):
        super(Decoder, self).__init__()
        channels = ch[::-1]
        num_res_blocks = num_res_blocks
        in_channels = channels[0]

        layers = [nn.Conv2d(z_channels, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  AtttnBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]
        
        #upsampling
        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(AttnBlock(in_channels))
            if i != 0:
                layers.append(UpSample(in_channels, resamp_with_conv))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, out_ch, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

        