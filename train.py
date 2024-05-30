import os
import datetime
import torch
import copy
import numpy as np
import warnings
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from scCNM_model import Create_Model, ce_loss, rec_loss
from data.data_process import xs_gen
from data import create_dataset
from sklearn.cluster import KMeans
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Used to specify which device to use to run the PyTorch model

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(opt):
    setup_seed(opt.seed)
    start_t = datetime.datetime.now()  # Get current time

    data = create_dataset(opt)            # Create dataset

    data_bk = copy.deepcopy(data)
    data_A_size = data.dataset.A_data.shape[0]
    data_B_size = data.dataset.B_data.shape[0]
    print('A的大小：', data_A_size)
    print('B的大小：', data_B_size)
    opt.A_col = data.dataset.A_data.shape[1]
    opt.B_col = data.dataset.B_data.shape[1]
    label_t_cn = np.loadtxt(opt.label, dtype='int32', delimiter=',')
    unique_K, counts_K = np.unique(label_t_cn, return_counts=True)
    opt.batch_size = data_A_size
    K_cluster = len(unique_K)

    model = Create_Model(opt) # Create Model
    optimizer_model = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98))

    epochs = []
    train_loss_rec_A = []
    train_loss_rec_B = []

    # Start training the model
    model.train()

    print("Iteration Start:")
    for epoch in tqdm(range(opt.epochs)): # Iteration Start
        data_train = copy.deepcopy(data_bk)

        for step, x in xs_gen(data_train, opt.batch_size, 1):
            if epoch % 1 == 0:
                epochs.append(epoch)
            real_A = x['A'].to(device)
            real_B = x['B'].to(device)
            real_A = real_A.unsqueeze(1)
            dna_enc, rna_enc, h_enc, rec_DNA, p_SNV = model(real_A, real_B)
            real_A = real_A.squeeze()

            optimizer_model.zero_grad()  # set model's gradients to zero

            loss_rec_A = rec_loss(rec_DNA, real_A)
            loss_rec_B = ce_loss(p_SNV, real_B)

            loss_model = loss_rec_A + loss_rec_B
            loss_model.backward()   # calculate gradients for model
            optimizer_model.step()

            if epoch % 1 ==0:
                train_loss_rec_A.append(loss_rec_A.data.cpu().numpy())
                train_loss_rec_B.append(loss_rec_B.data.cpu().numpy())

    # get latent representation of single cells after scCNM training is completed
    a = []
    data_eval = copy.deepcopy(data_bk)
    model.eval()
    for step, x in xs_gen(data_eval, opt.batch_size, 0):
        real_A_eval  = x['A'].to(device)
        real_B_eval = x['B'].to(device)
        real_A_eval = real_A_eval.unsqueeze(1)
        with torch.no_grad():
            d1, d2, h_enc, rec_DNA, p_SNV = model(real_A_eval, real_B_eval)
            z = h_enc.cpu().detach().numpy()
            a.append(z)

    for id, mu in enumerate(a):
        if id == 0:
            features = mu
        else:
            features = np.r_[features, mu]

    print("scCNM:")
    features_array = np.array(features)
    np.savetxt('./results/latent.txt', features_array,fmt='%.6f', delimiter=',') # Save features to TXT file
    print("latent.txt saved successfully")
    # use Gaussian mixture model to cluster the single cells
    print('clustering the cells...')

    warnings.simplefilter(action='ignore', category=FutureWarning)
    print("cluster =============")
    kmeans = KMeans(n_clusters=K_cluster, random_state=100)
    label_pred = kmeans.fit_predict(features)
    K_means_ARI = adjusted_rand_score(label_t_cn, label_pred)
    K_means_NMI = normalized_mutual_info_score(label_t_cn, label_pred)

    label_p_array = np.array(label_pred)
    label_p_array = label_p_array.reshape(1, -1)
    np.savetxt('./results/label.txt', label_p_array, fmt='%d',delimiter=',')  #Save label_p to TXT file
    print("label.txt saved successfully")
    print("cluster successfully")
    print("Performance Indicator：")
    print('cluster_num:', K_cluster, 'ARI: ', round(K_means_ARI, 6),'NMI: ', round(K_means_NMI, 6))


    end_t = datetime.datetime.now()
    print('elapsed time: ', (end_t - start_t).seconds)



