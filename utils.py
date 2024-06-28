""" helper function

author axiumao
"""

import sys
import torch

import numpy as np

from torch.utils.data import DataLoader

from dataset import My_Dataset_Train, My_Dataset_Valid, My_Dataset_Test

def get_network(args):
    """ return given network
    """

    if args.net == 'canet':
        from models.canet import canet
        net = canet(args)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_weighted_mydataloader(pathway, batch_size=16, num_workers=2, shuffle=True):
    Mydataset = My_Dataset_Train(pathway, transform=None)
    all_labels_h = [label_h.tolist() for _, _, _, label_h, _, _ in Mydataset]
    all_labels_s = [label_s.tolist() for _, _, _, _, label_s, _ in Mydataset]
    all_labels_c = [label_c.tolist() for _, _, _, _, _, label_c in Mydataset]
    
    number_h = np.unique(torch.Tensor(all_labels_h)[:,0], return_counts = True)[1]
    number_s = np.unique(torch.Tensor(all_labels_s)[:,0], return_counts = True)[1]
    number_c = np.unique(torch.Tensor(all_labels_c)[:,0], return_counts = True)[1]
    Data_loader = DataLoader(Mydataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return Data_loader, number_h, number_s, number_c


def get_mydataloader_valid(pathway, data_id = 1, batch_size=16, num_workers=2, shuffle=True):
    Mydataset = My_Dataset_Valid(pathway, data_id, transform=None)
    Data_loader = DataLoader(Mydataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return Data_loader


def get_mydataloader_test(pathway, data_id = 1, batch_size=16, num_workers=2, shuffle=True):
    Mydataset = My_Dataset_Test(pathway, data_id, transform=None)
    Data_loader = DataLoader(Mydataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return Data_loader