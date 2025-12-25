import torch
import torch.nn as nn
import torch.nn.functional as F

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)

class Convolution(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels= None, residual= False):
        super(Convolution, self).__init__()
        
        self.residual = residual
        if mid_channels is None:
            mid_channels = out_channels

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):

        if self.residual:
            return F.gelu(x + self.conv_layer(x))
        else:
            return self.conv_layer(x)

class Upsampling(nn.Module):

    def __init__(self, in_channels, out_channels, embedding= 256):
        super(Upsampling, self).__init__()

        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.layers = nn.Sequential(
            Convolution(in_channels, in_channels, residual= True),
            Convolution(in_channels, out_channels)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding, out_channels)
        )

    def forward(self, x, x_skip, t):

        x = self.up_sample(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.layers(x)
        t = self.emb_layer(t)[:,:, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + t

class DownSampling(nn.Module):

    def __init__(self, in_channels, out_channels, embedding= 256):
        super(DownSampling, self).__init__()

        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2),
            Convolution(in_channels, in_channels, residual= True),
            Convolution(in_channels, out_channels)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding, out_channels)
        )

    def forward(self, x, t):
        x = self.down_sample(x)
        t = self.emb_layer(t)[:,:, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + t
    

class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, tdim):
        super(UNet, self).__init__()

        self.tdim = tdim
        self.inc = Convolution(in_channels, 64)
        self.down1 = DownSampling(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = DownSampling(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = DownSampling(256, 256)
        self.sa3 = SelfAttention(256)

        self.bot1 = Convolution(256, 512)
        self.bot2 = Convolution(512, 512)
        self.bot3 = Convolution(512, 256)

        self.up1 = Upsampling(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Upsampling(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Upsampling(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.tdim)
        return self.unet_forwad(x, t)
