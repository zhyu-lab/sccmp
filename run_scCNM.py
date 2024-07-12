import argparse
from train import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="scCNM")
    parser.add_argument('--gpu', type=int, default=0, help='which GPU to use.')
    parser.add_argument('--epochs', type=int, default=150, help='number of epoches to train the scCNM.')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
    parser.add_argument('--latent_dim', type=int, default=50, help='the latent dimension.')
    parser.add_argument('--w', type=float, default=1000, help='the hyperparameter lambda.')
    parser.add_argument('--neighbors', type=int, default=5, help='number of neighbors for each cell to construct the cell graph.')
    parser.add_argument('--cnv', type=str, default='')
    parser.add_argument('--snv', type=str, default='')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    args = parser.parse_args()
    main(args)
