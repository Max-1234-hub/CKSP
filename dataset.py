""" train and test dataset

author axiumao
"""

import torch
from torch.utils.data import Dataset
   
class My_Dataset_Train(Dataset):

    def __init__(self, pathway, transform=None):
        X_train_h, _, _, Y_train_h, _, _, X_train_s, _, _, Y_train_s, _, _, X_train_c, _, _, Y_train_c, _, _ = torch.load(pathway)
        self.data_h, self.labels_h, self.data_s, self.labels_s, self.data_c, self.labels_c \
            = X_train_h, Y_train_h, X_train_s, Y_train_s, X_train_c, Y_train_c
        # self.data, self.labels = Data_Segm_random(df_data, n_persegm=40)
        #if transform is given, we transoform data using
        self.transform = transform

    def __len__(self):
        return len(self.data_h)

    def __getitem__(self, index):
        image_h, image_s, image_c = self.data_h[index], self.data_s[index], self.data_c[index] #troch.size: [1,200,6]
        label_h, label_s, label_c = self.labels_h[index], self.labels_s[index], self.labels_c[index]  #torch.size: [1]
        
        
        if self.transform:
            image_h, image_s, image_c = self.transform(image_h), self.transform(image_s), self.transform(image_c)

        return image_h, image_s, image_c, label_h, label_s, label_c


class My_Dataset_Valid(Dataset):

    def __init__(self, pathway, data_id, transform=None):
        _, X_valid_h, _, _, Y_valid_h, _, _, X_valid_s, _, _, Y_valid_s, _, _, X_valid_c, _, _, Y_valid_c, _ = torch.load(pathway)
        if data_id == 0:
            self.data, self.labels = X_valid_h, Y_valid_h
        elif data_id == 1:
            self.data, self.labels = X_valid_s, Y_valid_s
        else:
            self.data, self.labels = X_valid_c, Y_valid_c
        # self.data, self.labels = Data_Segm_random(df_data, n_persegm=40)
        #if transform is given, we transoform data using
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index] #troch.size: [1,200,6]
        label = self.labels[index]  #torch.size: [1]
        
        if self.transform:
            image = self.transform(image)

        return image, label


class My_Dataset_Test(Dataset):

    def __init__(self, pathway, data_id, transform=None):
        _, _, X_test_h, _, _, Y_test_h, _, _, X_test_s, _, _, Y_test_s, _, _, X_test_c, _, _, Y_test_c = torch.load(pathway)
        if data_id == 0:
            self.data, self.labels = X_test_h, Y_test_h
        elif data_id == 1:
            self.data, self.labels = X_test_s, Y_test_s
        else:
            self.data, self.labels = X_test_c, Y_test_c
        # self.data, self.labels = Data_Segm_random(df_data, n_persegm=40)
        #if transform is given, we transoform data using
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index] #troch.size: [1,200,6]
        label = self.labels[index]  #torch.size: [1]
        
        if self.transform:
            image = self.transform(image)

        return image, label
