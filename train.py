import os
import datetime
import torch
import copy
import numpy as np
import warnings

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter

from networks import scCMP, ce_loss_weighted, mse_loss
from graph_function import create_edge_index
from data import create_dataset
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    setup_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
    start_t = datetime.datetime.now()  # Get current time

    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.mkdir(args.output)

    data = create_dataset(args)            # Create dataset

    data_cn_size = data.dataset.A_data.shape[0]
    data_snv_size = data.dataset.B_data.shape[0]
    dim_cn = data.dataset.A_data.shape[1]
    dim_snv = data.dataset.B_data.shape[1]
    assert data_cn_size == data_snv_size
    print('size of CN data：', data.dataset.A_data.shape)
    print('size of SNV data：', data.dataset.B_data.shape)

    tmp = data.dataset.B_data.data.cpu().numpy()
    sparsity_ratio = np.sum(tmp < 0.5) / data_snv_size / dim_snv
    print('sparsity of SNV data：', sparsity_ratio)

    e, A = create_edge_index(data.dataset.A_data, args.neighbors)
    e = torch.from_numpy(e).to(device)

    label_t = np.loadtxt(args.label, dtype='int32', delimiter=',')
    u_ids, u_counts = np.unique(label_t, return_counts=True)
    num_cluster = len(u_ids)

    model = scCMP(dim_cn, dim_snv, args.latent_dim).to(device)  # Create scCMP Model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    train_loss_cn = []
    train_loss_snv = []
    train_loss = []
    epochs = []

    # Start training the model
    model.train()

    print("Model training start:")
    for epoch in range(args.epochs):
        epochs.append(epoch+1)
        data_train = copy.deepcopy(data)

        data_cn = data_train.dataset.A_data.to(device)
        data_snv = data_train.dataset.B_data.to(device)
        data_cn = data_cn.unsqueeze(1)
        enc_cn, enc_snv, h_enc, rec_cn, rec_snv, rec_A = model(data_cn, data_snv, e)
        data_cn = data_cn.squeeze()

        optimizer.zero_grad()  # set model's gradients to zero

        loss_rec_cn = mse_loss(rec_cn, data_cn)
        loss_rec_snv = ce_loss_weighted(rec_snv, data_snv, args.w)

        loss = loss_rec_cn + loss_rec_snv
        loss.backward()   # calculate gradients for model
        optimizer.step()

        train_loss_cn.append(loss_rec_cn.data.cpu().numpy())
        train_loss_snv.append(loss_rec_snv.data.cpu().numpy())
        train_loss.append(loss.data.cpu().numpy())
        print("epoch: " + str(epoch) + ", cn loss:" + str(loss_rec_cn.data.cpu().numpy()) +
              ", snv loss:" + str(loss_rec_snv.data.cpu().numpy()) +
              ", total loss:" + str(loss.data.cpu().numpy()))

    np.savetxt(output_dir + '/loss_cn.txt', np.c_[np.reshape(train_loss_cn, (1, len(train_loss_cn)))], fmt='%f', delimiter=',')
    np.savetxt(output_dir + '/loss_snv.txt', np.c_[np.reshape(train_loss_snv, (1, len(train_loss_snv)))], fmt='%f', delimiter=',')
    np.savetxt(output_dir + '/loss_total.txt', np.c_[np.reshape(train_loss, (1, len(train_loss)))], fmt='%f', delimiter=',')

    # get latent representation of single cells
    latent_features = []
    data_eval = copy.deepcopy(data)
    model.eval()
    data_cn = data_eval.dataset.A_data.to(device)
    data_snv = data_eval.dataset.B_data.to(device)
    data_cn = data_cn.unsqueeze(1)
    with torch.no_grad():
        enc_cn, enc_snv, h_enc, rec_cn, rec_snv, rec_A = model(data_cn, data_snv, e)
        latent_features = h_enc.cpu().detach().numpy()
        rec_snv = rec_snv.cpu().detach().numpy()

    latent_features = np.array(latent_features)
    np.savetxt(output_dir + '/latent.txt', latent_features, fmt='%.3f', delimiter=',')  # Save latents to file
    print("latent features saved successfully")
    rec_snv = np.array(rec_snv)
    np.savetxt(output_dir + '/rec-snv.txt', rec_snv, fmt='%.3f', delimiter=',')  # Save reconstructed snv data to file
    print("reconstructed snv data saved successfully")

    # use k-means to cluster the single cells
    print('clustering the cells...')
    warnings.simplefilter(action='ignore', category=FutureWarning)
    kmeans = KMeans(n_clusters=num_cluster, n_init=100)
    label_p = kmeans.fit_predict(latent_features)
    print(Counter(label_p))

    ari = adjusted_rand_score(label_t, label_p)
    nmi = normalized_mutual_info_score(label_t, label_p)
    print('cluster_num:', num_cluster, 'ARI: ', round(ari, 6), 'NMI: ', round(nmi, 6))

    label_p = np.array(label_p).reshape(1, -1)
    np.savetxt(output_dir + '/label.txt', label_p, fmt='%d', delimiter=',')  # Save labels to file
    print("cell labels saved successfully")

    end_t = datetime.datetime.now()
    print('elapsed time: ', (end_t - start_t).seconds)
