import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(net, init_type):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero_()

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

#This class implements a convolutional autoencoder
class module_1(nn.Module):
    def __init__(self, in_dim, z_dim, k_size):
        super(module_1, self).__init__()
        self.device = device
        d = in_dim
        for i in range(4):
            d = np.floor((d - k_size) / 1 + 1)
        d = np.int32(d)
        k_size_resid = (in_dim - d) + 1

        self.resid_conv_encoder = nn.Sequential(nn.Conv1d(1, 64, k_size_resid, stride=1),)
        self.resid_conv_decoder = nn.ConvTranspose1d(64, 1, k_size_resid, stride=1)
        self.act_fun = nn.LeakyReLU()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 512, k_size, stride=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(512, 256, k_size, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 128, k_size, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 64, k_size, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )

        self.fc1 = nn.Linear(d * 64, z_dim)
        self.fc2 = nn.Linear(z_dim, d * 64)
        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 128, k_size, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 256, k_size, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(256, 512, k_size, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(512, 1, k_size, stride=1),
        )
        init_weights(self, init_type='kaiming')

    def bottleneck(self, h):
        z = self.fc1(h)
        return z

    def encode_1(self, x):
        residual = self.resid_conv_encoder(x)
        h = self.encoder(x)
        h += residual
        h = self.act_fun(h)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z

    def decode_1(self, z):
        z = self.fc2(z)
        z = z.view(z.size(0), 64, int(z.size(1) / 64))
        residual = self.resid_conv_decoder(z)
        z = self.decoder(z)
        z += residual
        z = self.act_fun(z)
        z = z.squeeze()
        return z

    def forward(self, x):
        z = self.encode_1(x)
        y = self.decode_1(z)
        return z, y

class Encoder2(torch.nn.Module):
    """
    This class implements Encoder
    """
    def __init__(self, d_in, d_out):
        super(Encoder2, self).__init__()
        self.enc_layer1 = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        self.enc_layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )
        self.enc_layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.enc_layer4 = nn.Sequential(
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.enc_layer5 = nn.Sequential(
            nn.Linear(64, d_out),
        )

        init_weights(self, init_type='kaiming')

    def forward(self, x):
        x = self.enc_layer1(x)
        x = self.enc_layer2(x)
        x = self.enc_layer3(x)
        x = self.enc_layer4(x)
        x = self.enc_layer5(x)
        return x

class Decoder_2(nn.Module):
    """
    This class implements Decoder
    """
    def __init__(self, d_in, d_out):
        super(Decoder_2, self).__init__()
        self.dec_layer1 = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.LeakyReLU(),
        )
        self.dec_layer2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
        )
        self.dec_layer3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
        )
        self.dec_layer4 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(),
        )
        self.dec_layer5 = nn.Sequential(
            nn.Linear(512, d_out),
        )
        init_weights(self, init_type='kaiming')

    def forward(self, x):
        x = self.dec_layer1(x)
        x = self.dec_layer2(x)
        x = self.dec_layer3(x)
        x = self.dec_layer4(x)
        x = self.dec_layer5(x)
        return torch.sigmoid(x)

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

class AE_SNV(nn.Module):
    def __init__(self, encoder_B, Decoder_B):
        super(AE_SNV, self).__init__()
        self.encoder_2 = encoder_B
        self.decoder_2 = Decoder_B

    def forward(self, enc_inputs2):
        enc_outputs2 = self.encoder_2(enc_inputs2)
        rec_SNV = self.decoder_2(enc_outputs2)

        return  enc_outputs2, rec_SNV

class FC1(nn.Module):

    def __init__(self, z_emb_size1, dropout_rate):
        super(FC1, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(z_emb_size1, z_emb_size1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
        )
    def forward(self, z_x, z_y):
        q_x = self.fc1(z_x)
        q_y = self.fc1(z_y)
        return q_x, q_y

class con_att_scCNM(nn.Module):
    def __init__(self, module_1, AE_SNV, opt):
        super(con_att_scCNM, self).__init__()
        self.module_CNV = module_1
        self.AE_SNV = AE_SNV
        self.attlayer1 = SelfAttention(dropout=0.1)
        self.attlayer2 = SelfAttention(dropout=0.1)
        self.fc1 = FC1(opt.latent_dim, dropout_rate=0.1)
        self.fc2 = nn.Linear(2 * opt.latent_dim, opt.latent_dim)

    def forward(self,enc_inputs1,enc_inputs2):
        enc_outputs1 = self.module_CNV.encode_1(enc_inputs1)
        enc_outputs2 = self.AE_SNV.encoder_2(enc_inputs2)
        ## attention for omics specific information of CNV
        zx_weights, z_x = self.attlayer1(enc_outputs1, enc_outputs1, enc_outputs1)
        ## attention for omics specific information of SNV
        zy_weights, z_y = self.attlayer2(enc_outputs2, enc_outputs2, enc_outputs2)

        z_i, z_j = self.fc1(enc_outputs1,  enc_outputs2)
        h_enc = torch.cat([ z_i, z_j], dim=1)
        h_enc = self.fc2(h_enc)
        h_enc = z_x + z_y + h_enc

        rec_CNV = self.module_CNV.decode_1(h_enc)
        P_SNV = self.AE_SNV.decoder_2(h_enc)

        return enc_outputs1, enc_outputs2, h_enc, rec_CNV, P_SNV


