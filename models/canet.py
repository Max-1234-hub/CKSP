# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 15:41:25 2021

@author: axmao2-c
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CaNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.conv1 = BasicBlock_b(in_channels=1, out_channels=8, r=args.r, alpha=args.alpha, isLoRA=False, isLayer=False)
        self.conv2 = BasicBlock_b(in_channels=8, out_channels=16, r=args.r, alpha=args.alpha, isLoRA=True, isLayer=True)
        self.conv3 = BasicBlock_b(in_channels=16, out_channels=32, r=args.r, alpha=args.alpha, isLoRA=True, isLayer=True)
        self.conv4 = BasicBlock_b(in_channels=32, out_channels=64, r=args.r, alpha=args.alpha, isLoRA=True, isLayer=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))
        
        # self.conv5_acc = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=(1,1)),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))
        
        self.conv5_acc = nn.Conv2d(64, 64, kernel_size=(1,1))
        self.BN_h_5 = nn.BatchNorm2d(64)
        self.BN_s_5 = nn.BatchNorm2d(64)
        self.BN_c_5 = nn.BatchNorm2d(64)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_h = nn.Linear(64, 5)
        self.fc_s = nn.Linear(64, 3)
        self.fc_c = nn.Linear(64, 5)

    def forward(self, x, y):
        xa = x[:,:,:,0:3]
        xa = xa.permute(0,1,3,2)
        
        _, _, _, time_scale = xa.size() #size = [4,1,3,200]
        if time_scale != 50:  #统一输入到网络的数据大小；
            x_transformed = F.interpolate(xa, size = [3, 50], mode = 'bilinear', align_corners=True)
        else:           
            x_transformed = xa
        
        output_x = self.conv1(x_transformed, y)
        output_x = self.maxpool(output_x)
        output_x = self.conv2(output_x, y)
        output_x = self.maxpool(output_x)
        output_x = self.conv3(output_x, y)
        output_x = self.maxpool(output_x)
        output_x = self.conv4(output_x, y)
        # print(output_x.size())
    
        output_x = self.conv5_acc(output_x)
        if y[0][1] == 0:
            output_x = self.BN_h_5(output_x)
        elif y[0][1] == 1:
            output_x = self.BN_s_5(output_x)
        else:
            output_x = self.BN_c_5(output_x)
        
        output_x = nn.ReLU(inplace=True)(output_x)
        
        output_x = self.avg_pool(output_x) #[batch_size, num_filters, 1, 1]
        output_x = output_x.view(output_x.size(0), -1) #[batch_size, num_filters]
        
        if y[0][1] == 0:
            output = self.fc_h(output_x)
        elif y[0][1] == 1:
            output = self.fc_s(output_x)
        else:
            output = self.fc_c(output_x)
        
        # output = self.fc(output_x)
        # print(output.size())
    
        return output

class BasicBlock_b(nn.Module):
    def __init__(self, in_channels, out_channels, r, alpha, isLoRA=True, isLayer=True):
        super().__init__()

        self.isLoRA = isLoRA
        self.isLayer = isLayer
        #residual function
        self.residual_function_acc = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        
        self.BN_h_1 = nn.BatchNorm2d(out_channels)
        self.BN_s_1 = nn.BatchNorm2d(out_channels)
        self.BN_c_1 = nn.BatchNorm2d(out_channels)

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if self.isLayer == True:
            if self.isLoRA == True:
                self.shortcut_acc_h = LoRA_shortcut(in_channels, out_channels, kernel_size_1=1, kernel_size_2=3, r=r, lora_alpha=alpha)
                self.shortcut_acc_s = LoRA_shortcut(in_channels, out_channels, kernel_size_1=1, kernel_size_2=3, r=r, lora_alpha=alpha)
                self.shortcut_acc_c = LoRA_shortcut(in_channels, out_channels, kernel_size_1=1, kernel_size_2=3, r=r, lora_alpha=alpha)
            else:
                self.shortcut_acc_h = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1,1))
                self.shortcut_acc_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1,1))
                self.shortcut_acc_c = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1,1))
        else:
            self.shortcut_acc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1,1))
      
        self.BN_h_2 = nn.BatchNorm2d(out_channels)
        self.BN_s_2 = nn.BatchNorm2d(out_channels)
        self.BN_c_2 = nn.BatchNorm2d(out_channels)
        

    def forward(self, x_acc, y_acc):
        
        out_acc_res = self.residual_function_acc(x_acc)
        if y_acc[0][1] == 0:
            out_acc_res = self.BN_h_1(out_acc_res)
        elif y_acc[0][1] == 1:
            out_acc_res = self.BN_s_1(out_acc_res)
        else:
            out_acc_res = self.BN_c_1(out_acc_res)
        
        # out_acc_sho = self.shortcut_acc(x_acc)
        
        if y_acc[0][1] == 0:
            if self.isLayer == True:
                out_acc_sho = self.shortcut_acc_h(x_acc)
            else:
                out_acc_sho = self.shortcut_acc(x_acc)
            out_acc_sho = self.BN_h_2(out_acc_sho)
        elif y_acc[0][1] == 1:
            if self.isLayer == True:
                out_acc_sho = self.shortcut_acc_s(x_acc)
            else:
                out_acc_sho = self.shortcut_acc(x_acc)
            out_acc_sho = self.BN_s_2(out_acc_sho)
        else:
            if self.isLayer == True:
                out_acc_sho = self.shortcut_acc_c(x_acc)
            else:
                out_acc_sho = self.shortcut_acc(x_acc)
            out_acc_sho = self.BN_c_2(out_acc_sho)
        
        acc_output = nn.ReLU(inplace=True)(out_acc_res + out_acc_sho)
        
        return acc_output


class LoRA_shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_1, kernel_size_2, r, lora_alpha=1):
        super(LoRA_shortcut, self).__init__()
        self.lora_A = nn.Parameter(
            torch.zeros((r * kernel_size_1, in_channels * kernel_size_1)))
        self.lora_B = nn.Parameter(
          torch.zeros((out_channels * kernel_size_2, r * kernel_size_1)))
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.scaling = lora_alpha / r
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        
        kernel = (self.lora_B @ self.lora_A).view(self.out_channels, self.in_channels, self.kernel_size_1, self.kernel_size_2) * self.scaling
        x = F.conv2d(x, kernel, stride = (1,1), padding = (0,1))
        
        return x

def canet(args):

   return CaNet(args)


# import torch

# a = torch.randn(4,1,200,3)
# b = torch.Tensor([[4,0],[2,1],[2,2],[3,1]])
# net = canet()
# out = net(a,b)
