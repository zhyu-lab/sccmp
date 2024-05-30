import numpy as np
import torch
import math
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class data_set:
    def __init__(self,opt):
        self.opt = opt
        self.A_data = load_dataDNA(opt.cnv)  # load data from 'path'
        self.B_data = load_dataSNV(opt.snv) # load data from 'path'

def xs_gen(cell_list, batch_size, random1):
    data_copy = copy.deepcopy(cell_list.dataset)
    data1 = {'A': data_copy.A_data, 'B': data_copy.B_data}
    data_A_len = len(data_copy.A_data)
    steps = math.ceil(data_A_len / batch_size)

    if random1 == 1:
        data_copy.A_data = data_copy.A_data.numpy()
        data_copy.B_data = data_copy.B_data.numpy()
        np.random.shuffle(data_copy.A_data)
        np.random.shuffle(data_copy.B_data)
        data_copy.A_data = torch.from_numpy(data_copy.A_data)
        data_copy.B_data = torch.from_numpy(data_copy.B_data)
    elif random1 == 0:
        data_copy.A_data = data_copy.A_data.numpy()
        data_copy.B_data = data_copy.B_data.numpy()
        data_copy.A_data = torch.from_numpy(data_copy.A_data)
        data_copy.B_data = torch.from_numpy(data_copy.B_data)
    for i in range(steps):
        batch_x = data_copy.A_data[i * batch_size: i * batch_size + batch_size]
        batch_y = data_copy.B_data[i * batch_size: i * batch_size + batch_size]
        data1['A'] = batch_x
        data1['B'] = batch_y
        yield i, data1

def load_dataDNA(dir):
    data_o = np.loadtxt(dir, dtype='float32', delimiter=',')
    data_o = DNA_processing_log(data_o)
    return data_o

def load_dataSNV(dir):
    data_o = np.loadtxt(dir, dtype='float32', delimiter=',')
    return data_o

def DNA_processing_log(DNA_data):
    index = (DNA_data == 0.0)
    DNA_data[index] = 1
    data = np.log2(DNA_data)
    return data

