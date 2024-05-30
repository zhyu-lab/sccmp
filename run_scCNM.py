import argparse
from train import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="scCNM")
    parser.add_argument('--epochs', type=int, default=30, help='number of epoches to train the scCNM.')
    parser.add_argument('--lr', type=float, default=0.0006, help='learning rate.')
    parser.add_argument('--latent_dim', type=int, default=10, help='the latent dimension.')
    parser.add_argument('--A_col', type=int, help='CNV data dimensions')
    parser.add_argument('--B_col', type=int, help='SNV data dimensions')
    parser.add_argument('--cnv', type=str, default='./data/real/C/CHISEL_C_CN.txt')
    parser.add_argument('--snv', type=str, default='./data/real/C/CHISEL_C_SNV.txt')
    parser.add_argument('--label', type=str, default='./data/real/C/C_label.txt')
    parser.add_argument('--kernel_size', type=int, default=7, help='convolutional kernel size.')
    parser.add_argument('--seed', type=int, default=1, help='random seed.')
    opt = parser.parse_args()
    main(opt)