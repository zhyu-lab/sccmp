import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from torch_geometric.nn.conv import TAGConv
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath
import numpy as np


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )
        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.skip is not None:
            identity = self.skip(x)
        out = out + identity
        return out


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride),
            nn.LeakyReLU(inplace=False),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.skip is not None:
            identity = self.skip(x)
        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, upsample=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel_size, stride=2),
            nn.LeakyReLU(inplace=False),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )
        self.upsampling = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel_size, stride=2)
        )

    def forward(self, x):
        out = self.block(x)
        identity = self.upsampling(x)
        out += identity
        return out


class WMSA(nn.Module):
    """ Self-attention module in 1D Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, block_type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type = block_type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = nn.Parameter(self.relative_position_params.view(2*window_size-1, self.n_heads).transpose(0, 1))

    # def generate_mask(self, h, w, p, shift):
    def generate_mask(self, w, p, padding_size, shift):
        """ generating the mask of 1D SW-MSA
        Args:
            :param w: number of windows
            :param p: window size
            :param padding_size: padding size
            :param shift: shift parameters in CyclicShift.
        Returns:
            mask: should be (1 1 w p p),
        """
        mask = torch.zeros(w, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            s = p - padding_size
            mask[-1, :s, s:] = True
            mask[-1, s:, :s] = True
        else:
            if padding_size + shift <= p:
                s = p - padding_size - shift
                mask[-1, :s, s:] = True
                mask[-1, s:s+padding_size, :s] = True
                mask[-1, s:s+padding_size, s+padding_size:] = True
                mask[-1, s+padding_size:, :s+padding_size] = True
            else:
                t = p - shift
                s = p - (padding_size-t)
                mask[-1, :t, t:] = True
                mask[-1, t:, :t] = True
                mask[-2, :s, s:] = True
                mask[-2, s:, :s] = True
        mask = rearrange(mask, 'w p1 p2 -> 1 1 w p1 p2')
        return mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b n c]
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b n c]
        """

        original_n = x.size(1)
        remainder = original_n % self.window_size
        needs_padding = remainder > 0
        padding_size = 0
        if needs_padding:
            padding_size = self.window_size - remainder
            x = F.pad(x, (0, 0, 0, padding_size, 0, 0), value=0)

        if self.type != 'W':
            x = torch.roll(x, shifts=-(self.window_size // 2), dims=1)
        x = rearrange(x, 'b (w p) c -> b w p c', p=self.window_size)
        windows = x.size(1)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b w p (threeh c) -> threeh b w p c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')

        attn_mask = self.generate_mask(windows, self.window_size, padding_size, shift=self.window_size // 2)
        sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b w p c -> b (w p) c', w=windows, p=self.window_size)

        if self.type != 'W':
            output = torch.roll(output, shifts=self.window_size // 2, dims=1)
        return output[:, :original_n]

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i] for i in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:, :, 0].long()]


class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, block_type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(SwinBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert block_type in ['W', 'SW']
        self.type = block_type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W'):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        self.trans_block = SwinBlock(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type)
        self.conv1_1 = nn.Conv1d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv1d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), [self.conv_dim, self.trans_dim], dim=1)
        conv_x = self.conv_block(conv_x)
        trans_x = Rearrange('b c n -> b n c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b n c -> b c n')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x


class Autoencoder_CN(nn.Module):
    def __init__(self, in_dim, z_dim=3, n_channels=96, kernel_size=7, config=None, head_dim=None, drop_path_rate=0):
        super(Autoencoder_CN, self).__init__()
        self.z_dim = z_dim
        self.window_size = 49
        dim = n_channels
        if config is None:
            config = [2, 2, 2, 2, 2, 2]
        self.config = config
        if head_dim is None:
            head_dim = [32, 32, 32, 32, 32, 32]
        self.head_dim = head_dim

        d = in_dim
        k_sizes = []
        for i in range(4):
            if d % 2 == 0:
                d = np.int32((d - kernel_size + 1) / 2 + 1)
                k_sizes.append(kernel_size - 1)
            else:
                d = np.int32((d - kernel_size) / 2 + 1)
                k_sizes.append(kernel_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        self.m_down1 = [ConvTransBlock(dim, dim, self.head_dim[0], self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW')
                        for i in range(config[0])] + [nn.Conv1d(2 * dim, 2 * dim, kernel_size=k_sizes[1], stride=2)]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim[1], self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW')
                        for i in range(config[1])] + [nn.Conv1d(2 * dim, 2 * dim, kernel_size=k_sizes[2], stride=2)]
        self.m_down3 = [ConvTransBlock(dim, dim, self.head_dim[2], self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW')
                        for i in range(config[2])] + [nn.Conv1d(2 * dim, 1, kernel_size=k_sizes[3], stride=2)]

        self.m_up1 = [ConvTransBlock(dim, dim, self.head_dim[3], self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW')
                      for i in range(config[3])] + [nn.ConvTranspose1d(2 * dim, 2 * dim, kernel_size=k_sizes[2], stride=2)]
        self.m_up2 = [ConvTransBlock(dim, dim, self.head_dim[4], self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW')
                      for i in range(config[4])] + [nn.ConvTranspose1d(2 * dim, 2 * dim, kernel_size=k_sizes[1], stride=2)]
        self.m_up3 = [ConvTransBlock(dim, dim, self.head_dim[5], self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW')
                      for i in range(config[5])]

        self.encoder = nn.Sequential(*[ResidualBlockWithStride(1, 2 * dim, k_sizes[0], 2)] + self.m_down1 + self.m_down2 + self.m_down3)
        self.decoder = nn.Sequential(*[ResidualBlockUpsample(1, 2 * dim, k_sizes[3], 2)] + self.m_up1 + self.m_up2 + self.m_up3)

        self.latent_layer = nn.Linear(d, z_dim)
        self.m_up0 = nn.Sequential(
            nn.Linear(z_dim, d),
            nn.LeakyReLU()
        )

        self.ct1 = nn.Sequential(
            nn.ConvTranspose1d(2 * dim, 1, kernel_size=k_sizes[0], stride=2)
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.latent_layer(h)
        return z

    def decode(self, z):
        z = self.m_up0(z)
        z = z.view(z.size(0), 1, -1)
        z = self.decoder(z)
        y = self.ct1(z).squeeze()
        return y


# Define autoencoder for SNV data
class Autoencoder_SNV(nn.Module):
    def __init__(self, in_dim, z_dim=3):
        super(Autoencoder_SNV, self).__init__()
        self.z_dim = z_dim

        self.encoder = TAGConv(in_dim, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, in_dim)
        )

    def encode(self, x, e):
        return self.encoder(x, e)

    def decode(self, z):
        x = self.decoder(z)
        A = torch.matmul(z, torch.transpose(z, 0, 1))
        return torch.sigmoid(x), torch.sigmoid(A)


class SelfAttention(nn.Module):
    """
    attention
    """
    def __init__(self, dropout):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):

        queries = q
        keys = k
        values = v
        n, d = queries.shape
        scores = torch.mm(queries, keys.t()) / math.sqrt(d)
        att_weights = F.softmax(scores, dim=1)
        att_emb = torch.mm(self.dropout(att_weights), values)
        return att_weights, att_emb


# Define scCMP model
class scCMP(nn.Module):
    def __init__(self, in_dim_cn, in_dim_snv, z_dim, n_channels=32):
        super(scCMP, self).__init__()
        self.ae_cn = Autoencoder_CN(in_dim_cn, z_dim, n_channels)
        self.ae_snv = Autoencoder_SNV(in_dim_snv, z_dim)
        self.attlayer1 = SelfAttention(dropout=0.1)
        self.attlayer2 = SelfAttention(dropout=0.1)
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
        )
        self.fc2 = nn.Linear(2 * z_dim, z_dim)

    def forward(self, x, y, e):
        enc_cn = self.ae_cn.encode(x)
        enc_snv = self.ae_snv.encode(y, e)

        z_x = enc_cn
        z_y = enc_snv

        z_i = self.fc1(z_x)
        z_j = self.fc1(z_y)
        h_enc = torch.cat([z_i, z_j], dim=1)
        z_xy = self.fc2(h_enc)
        h_enc = z_x + z_y + z_xy

        rec_cn = self.ae_cn.decode(h_enc)
        rec_snv, A = self.ae_snv.decode(h_enc)

        return enc_cn, enc_snv, h_enc, rec_cn, rec_snv, A


# Define loss function
def ce_loss(predict, target):
    loss = torch.nn.BCELoss()
    return loss(predict, target)


def ce_loss_weighted(predict, target, w):
    tmp = w * target * torch.log(predict+1e-10) + (1-target) * torch.log(1-predict+1e-10)
    return -torch.mean(tmp)


def mse_loss(predict, target):
    loss = torch.nn.MSELoss()
    return loss(predict, target)
