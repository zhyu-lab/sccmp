import torch
from networks import  Encoder2, Decoder_2, module_1, AE_SNV, con_att_scCNM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Create_Model(opt): # define and create scCNM architecture
    A_col = opt.A_col
    B_col = opt.B_col
    encoder_B = Encoder2(B_col, opt.latent_dim).to(device)
    decoder_B = Decoder_2(opt.latent_dim, B_col).to(device)
    net_CNV = module_1(A_col, opt.latent_dim, opt.kernel_size).to(device)
    net_SNV = AE_SNV(encoder_B, decoder_B).to(device)
    net_att_gan_ae = con_att_scCNM(net_CNV, net_SNV, opt).to(device)

    return  net_att_gan_ae

# Define loss function
def ce_loss(predict, target):
    loss = torch.nn.BCELoss()
    return loss(predict, target)

def rec_loss(predict, target):
    loss = torch.nn.MSELoss()
    return loss(predict, target)








