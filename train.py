# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author axiumao
"""

import os
import csv
import argparse
import time
import numpy as np
import platform
import matplotlib as mpl
mpl.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
import random
# import torch.backends.cudnn as cudnn
# import matplotlib.font_manager as font_manager

import torch
import torch.nn as nn
import torch.optim as optim
from Class_balanced_loss import CB_loss

from conf import settings
from Regularization import Regularization
from utils import get_network, get_weighted_mydataloader, get_mydataloader_valid, get_mydataloader_test
from sklearn.metrics import f1_score, classification_report, confusion_matrix, recall_score, precision_score



def train(train_loader, network, optimizer, epoch, loss_function, samples_per_cls_h, samples_per_cls_s, samples_per_cls_c):

    start = time.time()
    network.train()
    train_acc_process, train_acc_process_h, train_acc_process_s, train_acc_process_c = [], [], [], []
    train_loss_process = []
    for batch_index, (images_h, images_s, images_c, labels_h, labels_s, labels_c) in enumerate(train_loader):

        if args.gpu:
            labels_h, labels_s, labels_c = labels_h.cuda(), labels_s.cuda(), labels_c.cuda()
            images_h, images_s, images_c = images_h.cuda(), images_s.cuda(), images_c.cuda()
            #loss_function = loss_function.cuda()
        # print("label",labels)
        
        optimizer.zero_grad() # clear gradients for this training step
        outputs_h = network(images_h, labels_h)
        outputs_s = network(images_s, labels_s)
        outputs_c = network(images_c, labels_c)
        loss_type = "focal"
        loss_cb_h = CB_loss(labels_h[:,0].to(torch.int64), outputs_h, samples_per_cls_h, 5,loss_type, args.beta, args.gamma)
        loss_cb_s = CB_loss(labels_s[:,0].to(torch.int64), outputs_s, samples_per_cls_s, 3,loss_type, args.beta, args.gamma)
        loss_cb_c = CB_loss(labels_c[:,0].to(torch.int64), outputs_c, samples_per_cls_c, 5,loss_type, args.beta, args.gamma)
        
        # loss_ce = loss_function(outputs, labels)
        loss = (loss_cb_h + loss_cb_s + loss_cb_c) / 3 # class-balanced focal loss (CMI-Net+CB focal loss)
        if args.weight_d > 0:
            loss = loss + reg_loss(net)
        
        loss.backward() # backpropogation, compute gradients
        optimizer.step() # apply gradients
        _, preds_h = outputs_h.max(1)
        _, preds_s = outputs_s.max(1)
        _, preds_c = outputs_c.max(1)
        # print("prediction",preds)
        correct_n_h = preds_h.eq(labels_h[:,0]).sum()
        correct_n_s = preds_s.eq(labels_s[:,0]).sum()
        correct_n_c = preds_c.eq(labels_c[:,0]).sum() 
        accuracy_iter_h = correct_n_h.float() / len(labels_h)
        accuracy_iter_s = correct_n_s.float() / len(labels_s)
        accuracy_iter_c = correct_n_c.float() / len(labels_c)
        accuracy_iter = (accuracy_iter_h + accuracy_iter_s + accuracy_iter_c) / 3
        
        if args.gpu:
            accuracy_iter_h = accuracy_iter_h.cpu()
            accuracy_iter_s = accuracy_iter_s.cpu()
            accuracy_iter_c = accuracy_iter_c.cpu()
            accuracy_iter = accuracy_iter.cpu()
        
        train_acc_process_h.append(accuracy_iter_h.numpy().tolist())
        train_acc_process_s.append(accuracy_iter_s.numpy().tolist())
        train_acc_process_c.append(accuracy_iter_c.numpy().tolist())
        train_acc_process.append(accuracy_iter.numpy().tolist())
        
        train_loss_process.append(loss.item())

    print('Training Epoch: {epoch} [{total_samples}]\tTrain_accuracy_total: {:.4f}\tTrain_accuracy_h: {:.4f}\tTrain_accuracy_s: {:.4f}\tTrain_accuracy_c: {:.4f}\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            np.mean(train_loss_process),
            np.mean(train_acc_process_h),
            np.mean(train_acc_process_s),
            np.mean(train_acc_process_c),
            np.mean(train_loss_process),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            total_samples=(len(train_loader.dataset)*3)
    ))
    
    Train_Accuracy_h.append(np.mean(train_acc_process_h))
    Train_Accuracy_s.append(np.mean(train_acc_process_s))
    Train_Accuracy_c.append(np.mean(train_acc_process_c))
    Train_Loss.append(np.mean(train_loss_process))
    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    
    return network


@torch.no_grad()
def eval_training(valid_loader_h, valid_loader_s, valid_loader_c, network,loss_function, epoch=0):

    start = time.time()
    network.eval()
    
    n_h, n_s, n_c = 0, 0, 0
    valid_loss_h, valid_loss_s, valid_loss_c = 0.0, 0.0, 0.0 # cost function error
    correct_h, correct_s, correct_c = 0.0, 0.0, 0.0
    class_target_h, class_target_s, class_target_c =[], [], []
    class_predict_h, class_predict_s, class_predict_c = [], [], []
    
    for i, loader_i in enumerate([valid_loader_h, valid_loader_s, valid_loader_c]):
        for (images, labels) in loader_i:
            if args.gpu:
                labels = labels.cuda()
                images = images.cuda()
                loss_function = loss_function.cuda()
    
            outputs = network(images, labels)
            loss = loss_function(outputs, labels[:,0].to(torch.int64))
            _, preds = outputs.max(1)
            
            if args.gpu:
                labels = labels.cpu()
                preds = preds.cpu()
            
            if i ==0:
                valid_loss_h += loss.item()
                correct_h += preds.eq(labels[:,0]).sum()
                class_target_h.extend(labels[:,0].numpy().tolist())
                class_predict_h.extend(preds.numpy().tolist())
                n_h +=1
            elif i==1:
                valid_loss_s += loss.item()
                correct_s += preds.eq(labels[:,0]).sum()
                class_target_s.extend(labels[:,0].numpy().tolist())
                class_predict_s.extend(preds.numpy().tolist())
                n_s +=1
            else:
                valid_loss_c += loss.item()
                correct_c += preds.eq(labels[:,0]).sum()
                class_target_c.extend(labels[:,0].numpy().tolist())
                class_predict_c.extend(preds.numpy().tolist())
                n_c +=1
         
    finish = time.time()
    
    ###Loss and accuracy
    loss_h, loss_s, loss_c = valid_loss_h / n_h, valid_loss_s / n_s, valid_loss_c / n_c
    acc_h, acc_s, acc_c = correct_h.float() / len(valid_loader_h.dataset), correct_s.float() / len(valid_loader_s.dataset), correct_c.float() / len(valid_loader_c.dataset) 
    
    print('Evaluating Network.....')
    print('Valid set: Epoch: {}, Average loss_h: {:.4f}, Average loss_s: {:.4f}, Average loss_c: {:.4f}, Accuracy_h: {:.4f}, Accuracy_s: {:.4f}, Accuracy_c: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch, loss_h, loss_s, loss_c, acc_h, acc_s, acc_c, finish - start))
    
    #Obtain f1_score of the prediction
    fs_h = f1_score(class_target_h, class_predict_h, average='macro')
    print('f1 score of horse dataset = {}'.format(fs_h))
    fs_s = f1_score(class_target_s, class_predict_s, average='macro')
    print('f1 score of sheep dataset = {}'.format(fs_s))
    fs_c = f1_score(class_target_c, class_predict_c, average='macro')
    print('f1 score of cattle dataset = {}'.format(fs_c))
    
    #Output the classification report
    print('------------')
    print('Classification Report of horse dataset')
    print(classification_report(class_target_h, class_predict_h))
    print('Classification Report of sheep dataset')
    print(classification_report(class_target_s, class_predict_s))
    print('Classification Report of cattle dataset')
    print(classification_report(class_target_c, class_predict_c))
    
    
    f1_s_h.append(fs_h)
    f1_s_s.append(fs_s)
    f1_s_c.append(fs_c)
    Valid_Loss.append((loss_h + loss_s + loss_c)/3)
    Valid_Accuracy_h.append(acc_h)
    Valid_Accuracy_s.append(acc_s)
    Valid_Accuracy_c.append(acc_c)
    
    print('Setting: Epoch: {}, Batch size: {}, Learning rate: {:.6f}, gpu:{}, seed:{}'.format(args.epoch, args.b, args.lr, args.gpu, args.seed))

    return acc_h, acc_s, acc_c, fs_h, fs_s, fs_c
        

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='canet', help='net type')
    parser.add_argument('--gpu', type = int, default=0, help='use gpu or not')
    parser.add_argument('--b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epoch',type=int, default=100, help='total training epoches')
    parser.add_argument('--seed',type=int, default=10, help='seed')
    parser.add_argument('--gamma',type=float, default=0, help='the gamma of focal loss')
    parser.add_argument('--beta',type=float, default=0.9999, help='the beta of class balanced loss')
    parser.add_argument('--weight_d',type=float, default=0.1, help='weight decay for regularization')
    parser.add_argument('--save_path',type=str, default='setting0', help='saved path of each setting')
    parser.add_argument('--r',type=int, default=8, help='r in LoRA')
    parser.add_argument('--alpha',type=int, default=16, help='alpha in LoRA')
    parser.add_argument('--data_path',type=str, default='C:\\Users\\axmao2-c\\OneDrive - City University of Hong Kong\\Desktop\\6J_Wearable sensors\\Datasets\\Total_data\\input_combined_data\\myTensor_acc_combined_1.pt', help='saved path of input data')
    args = parser.parse_args()
    
    setup_seed(args.seed)
    net = get_network(args)
    print(net)
    print('Setting: Epoch: {}, Batch size: {}, Learning rate: {:.6f}, gpu:{}, seed:{}'.format(args.epoch, args.b, args.lr, args.gpu, args.seed))

    sysstr = platform.system()
    if(sysstr =="Windows"):
        num_workers = 0
    else:
        num_workers = 8
        
    pathway = args.data_path
    if sysstr=='Linux': 
        pathway = args.data_path
    
    ###load the datasets.
    train_loader, number_train_h, number_train_s, number_train_c = \
        get_weighted_mydataloader(pathway, batch_size=args.b, num_workers=num_workers, shuffle=True)
    
    valid_loader_h = get_mydataloader_valid(pathway, data_id=0, batch_size=args.b, num_workers=num_workers, shuffle=True)
    valid_loader_s = get_mydataloader_valid(pathway, data_id=1, batch_size=args.b, num_workers=num_workers, shuffle=True)
    valid_loader_c = get_mydataloader_valid(pathway, data_id=2, batch_size=args.b, num_workers=num_workers, shuffle=True)
    
    test_loader_h = get_mydataloader_test(pathway, data_id=0, batch_size=args.b, num_workers=num_workers, shuffle=True)
    test_loader_s = get_mydataloader_test(pathway, data_id=1, batch_size=args.b, num_workers=num_workers, shuffle=True)
    test_loader_c = get_mydataloader_test(pathway, data_id=2, batch_size=args.b, num_workers=num_workers, shuffle=True)
    
    if args.weight_d > 0:
        reg_loss=Regularization(net, args.weight_d, p=2)
    else:
        print("no regularization")
    
    loss_function_CE = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.save_path, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path_pth = os.path.join(checkpoint_path, '{net}-{type}.pth')

    best_acc_h, best_acc_s, best_acc_c = 0.0, 0.0, 0.0
    Train_Loss = []
    Valid_Loss = []
    Train_Accuracy_h, Train_Accuracy_s, Train_Accuracy_c = [], [], []
    Valid_Accuracy_h, Valid_Accuracy_s, Valid_Accuracy_c = [], [], []
    f1_s_h, f1_s_s, f1_s_c = [], [], []
    best_epoch_h, best_epoch_s, best_epoch_c = 1, 1, 1
    best_weights_path_h = checkpoint_path_pth.format(net=args.net, type='best_h')
    best_weights_path_s = checkpoint_path_pth.format(net=args.net, type='best_s')
    best_weights_path_c = checkpoint_path_pth.format(net=args.net, type='best_c')
    # validation_loss = 0
    for epoch in range(1, args.epoch + 1):
        train_scheduler.step(epoch)
            
        net = train(train_loader, net, optimizer, epoch, loss_function_CE, number_train_h, number_train_s, number_train_c)
        acc_h, acc_s, acc_c, fs_valid_h, fs_valid_s, fs_valid_c = eval_training(valid_loader_h, valid_loader_s, valid_loader_c, net, loss_function_CE, epoch)

        #start to save best performance model (according to the accuracy on validation dataset) after learning rate decay to 0.01
        if epoch > settings.MILESTONES[0]:
            if best_acc_h < acc_h:
                best_acc_h = acc_h
                best_epoch_h = epoch
                torch.save(net.state_dict(), best_weights_path_h)
            if best_acc_s < acc_s:
                best_acc_s = acc_s
                best_epoch_s = epoch
                torch.save(net.state_dict(), best_weights_path_s)
            if best_acc_c < acc_c:
                best_acc_c = acc_c
                best_epoch_c = epoch
                torch.save(net.state_dict(), best_weights_path_c)
    print('best epoch of h, s, c is {}, {}, {}'.format(best_epoch_h, best_epoch_s, best_epoch_c))
    
    
    #plot accuracy varying over time
    font_1 = {'weight' : 'normal', 'size'   : 20}
    fig1=plt.figure(figsize=(12,9))
    plt.title('Accuracy',font_1)
    index_train = list(range(1,len(Train_Accuracy_h)+1))
    plt.plot(index_train,Train_Accuracy_h,color='skyblue',linestyle='-',linewidth=2,label='train_acc_h')
    plt.plot(index_train,Valid_Accuracy_h,color='red',linestyle='--',linewidth=2,label='valid_acc_h')
    plt.plot(index_train,Train_Accuracy_s,color='green',linestyle='-',linewidth=2,label='train_acc_s')
    plt.plot(index_train,Valid_Accuracy_s,color='cyan',linestyle='--',linewidth=2,label='valid_acc_s')
    plt.plot(index_train,Train_Accuracy_c,color='yellow',linestyle='-',linewidth=2,label='train_acc_c')
    plt.plot(index_train,Valid_Accuracy_c,color='magenta',linestyle='--',linewidth=2,label='valid_acc_c')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Accuracy',font_1)
    
    acc_figuresavedpath = os.path.join(checkpoint_path,'Accuracy_curve.png')
    plt.savefig(acc_figuresavedpath)
    # plt.show()
    
    #plot loss varying over time
    fig2=plt.figure(figsize=(12,9))
    plt.title('Loss',font_1)
    index_valid = list(range(1,len(Valid_Loss)+1))
    plt.plot(index_valid,Train_Loss,color='skyblue', label='train_loss')
    plt.plot(index_valid,Valid_Loss,color='red', label='valid_loss')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Loss',font_1)

    loss_figuresavedpath = os.path.join(checkpoint_path,'Loss_curve.png')
    plt.savefig(loss_figuresavedpath)
    # plt.show()
    
    #plot f1 score varying over time
    fig3=plt.figure(figsize=(12,9))
    plt.title('F1-score',font_1)
    index_fs = list(range(1,len(f1_s_h)+1))
    plt.plot(index_fs,f1_s_h,color='skyblue',label='valid_fs_h')
    plt.plot(index_fs,f1_s_s,color='red',label='valid_fs_s')
    plt.plot(index_fs,f1_s_c,color='green',label='valid_fs_c')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Loss',font_1)

    fs_figuresavedpath = os.path.join(checkpoint_path,'F1-score.png')
    plt.savefig(fs_figuresavedpath)
    # plt.show()
    
    out_txtsavedpath = os.path.join(checkpoint_path,'output.txt')
    f = open(out_txtsavedpath, 'w+')
    
    print('Setting: Seed:{}, Epoch: {}, Batch size: {}, Learning rate: {:.6f}, Weight decay: {}, gpu:{}, Data path: {}, Saved path: {}'.format(
        args.seed, args.epoch, args.b, args.lr, args.weight_d, args.gpu, args.data_path, args.save_path),
        file=f)

    ######load the best trained model and test testing data
    best_net_h, best_net_s, best_net_c = get_network(args), get_network(args), get_network(args)
    best_net_h.load_state_dict(torch.load(best_weights_path_h))
    best_net_s.load_state_dict(torch.load(best_weights_path_s))
    best_net_c.load_state_dict(torch.load(best_weights_path_c))
    
    # total_num_paras, trainable_num_paras = get_parameter_number(best_net)
    # print('The total number of network parameters = {}'.format(total_num_paras), file=f)
    # print('The trainable number of network parameters = {}'.format(trainable_num_paras), file=f)
    
    best_net_h.eval()
    best_net_s.eval()
    best_net_c.eval()
    number_h, number_s, number_c = 0, 0, 0
    correct_test_h, correct_test_s, correct_test_c = 0.0, 0.0, 0.0
    test_target_h, test_target_s, test_target_c =[], [], []
    test_predict_h, test_predict_s, test_predict_c = [], [], []
    
    with torch.no_grad():
        
        start = time.time()
        
        for i, loader_i in enumerate([test_loader_h, test_loader_s, test_loader_c]):
            for n_iter, (image, labels) in enumerate(loader_i):
                if n_iter == 1: 
                    print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(loader_i)))
    
                if args.gpu:
                    image = image.cuda()
                    labels = labels.cuda()
                    
                if i ==0:
                    output = best_net_h(image, labels)
                    output = torch.softmax(output, dim= 1)
                    preds = torch.argmax(output, dim =1)
                    
                    if args.gpu:
                        labels = labels.cpu()
                        preds = preds.cpu()
            
                    correct_test_h += preds.eq(labels[:,0]).sum()
                    test_target_h.extend(labels[:,0].numpy().tolist())
                    test_predict_h.extend(preds.numpy().tolist())
                    number_h +=1
                elif i ==1:
                    output = best_net_s(image, labels)
                    output = torch.softmax(output, dim= 1)
                    preds = torch.argmax(output, dim =1)
                    
                    if args.gpu:
                        labels = labels.cpu()
                        preds = preds.cpu()
                    
                    correct_test_s += preds.eq(labels[:,0]).sum()
                    test_target_s.extend(labels[:,0].numpy().tolist())
                    test_predict_s.extend(preds.numpy().tolist())
                    number_s +=1
                else:
                    output = best_net_c(image, labels)
                    output = torch.softmax(output, dim= 1)
                    preds = torch.argmax(output, dim =1)
                    
                    if args.gpu:
                        labels = labels.cpu()
                        preds = preds.cpu()
                        
                    correct_test_c += preds.eq(labels[:,0]).sum()
                    test_target_c.extend(labels[:,0].numpy().tolist())
                    test_predict_c.extend(preds.numpy().tolist())
                    number_c +=1

        finish = time.time()
        acc_test_h = correct_test_h.float() / len(test_loader_h.dataset)
        acc_test_s = correct_test_s.float() / len(test_loader_s.dataset)
        acc_test_c = correct_test_c.float() / len(test_loader_c.dataset)
        total_accuracy_test = (acc_test_h + acc_test_s + acc_test_c) / 3
        print('Testing network......', file=f)
        print('Test set: Accuracy_h: {:.5f}, Accuracy_s: {:.5f}, Accuracy_c: {:.5f}, Total_Accuracy: {:.5f}, Time consumed: {:.5f}s'.format(
            acc_test_h, acc_test_s, acc_test_c, total_accuracy_test, finish - start), file=f)
        
        #Obtain f1_score of the prediction
        fs_test_h = f1_score(test_target_h, test_predict_h, average='macro')
        print('f1 score_h = {:.5f}'.format(fs_test_h), file=f)
        fs_test_s = f1_score(test_target_s, test_predict_s, average='macro')
        print('f1 score_s = {:.5f}'.format(fs_test_s), file=f)
        fs_test_c = f1_score(test_target_c, test_predict_c, average='macro')
        print('f1 score_c = {:.5f}'.format(fs_test_c), file=f)
        
        prec_test_h = precision_score(test_target_h, test_predict_h, average='macro')
        print('precision_h = {:.5f}'.format(prec_test_h), file=f)
        prec_test_s = precision_score(test_target_s, test_predict_s, average='macro')
        print('precision_s = {:.5f}'.format(prec_test_s), file=f)
        prec_test_c = precision_score(test_target_c, test_predict_c, average='macro')
        print('precision_c = {:.5f}'.format(prec_test_c), file=f)
        
        recall_test_h = recall_score(test_target_h, test_predict_h, average='macro')
        print('recall_h = {:.5f}'.format(recall_test_h), file=f)
        recall_test_s = recall_score(test_target_s, test_predict_s, average='macro')
        print('recall_s = {:.5f}'.format(recall_test_s), file=f)
        recall_test_c = recall_score(test_target_c, test_predict_c, average='macro')
        print('recall_c = {:.5f}'.format(recall_test_c), file=f)
        
        #Output the classification report
        print('------------', file=f)
        print('Classification Report of horse dataset', file=f)
        print(classification_report(test_target_h, test_predict_h), file=f)
        print('------------', file=f)
        print('Classification Report of sheep dataset', file=f)
        print(classification_report(test_target_s, test_predict_s), file=f)
        print('------------', file=f)
        print('Classification Report of cattle dataset', file=f)
        print(classification_report(test_target_c, test_predict_c), file=f)
        
        # print('Label values: {}'.format(test_target), file=f)
        # print('Predicted values: {}'.format(test_predict), file=f)
        
        label_results_path = os.path.join('label_results', args.net, settings.TIME_NOW)
        
        #create checkpoint folder to save label results;
        if not os.path.exists(label_results_path):
            os.makedirs(label_results_path)
        label_file_name = args.save_path + '.csv'
        label_results_path_name = os.path.join(label_results_path, label_file_name)
        label_f = open(label_results_path_name, 'w+')
        
        print(test_target_h, file=label_f)
        print(test_predict_h, file=label_f)
        print(test_target_s, file=label_f)
        print(test_predict_s, file=label_f)
        print(test_target_c, file=label_f)
        print(test_predict_c, file=label_f)
        
        if not os.path.exists('./results.csv'):
            with open("./results.csv", 'w+') as csvfile:
                writer_csv = csv.writer(csvfile)
                writer_csv.writerow(['index','accuracy_h','f1-score_h','precision_h','recall_h', \
                                     'accuracy_s','f1-score_s','precision_s','recall_s', \
                                     'accuracy_c','f1-score_c','precision_c','recall_c', 'time_consumed'])
        
        with open("./results.csv", 'a+') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow([args.seed, acc_test_h, fs_test_h, prec_test_h, recall_test_h, \
                                 acc_test_s, fs_test_s, prec_test_s, recall_test_s, \
                                 acc_test_c, fs_test_c, prec_test_c, recall_test_c,finish-start])
        
        Class_labels_h = ['Grazing', 'Galloping', 'Standing', 'Trotting', 'Walking']
        Class_labels_s = ['Grazing', 'Active', 'Inactive']
        Class_labels_c = ['Grazing', 'Moving', 'Resting', 'Ruminating', 'Salting']
        #Show the confusion matrix so that it can help us observe the results more intuitively
        def show_confusion_matrix(validations, predictions, Class_labels, species):
            matrix = confusion_matrix(validations, predictions) #No one-hot
            #matrix = confusion_matrix(validations.argmax(axis=1), predictions.argmax(axis=1)) #One-hot
            plt.figure(figsize=(6, 4))
            sns.heatmap(matrix,
                  cmap="coolwarm",
                  linecolor='white',
                  linewidths=1,
                  xticklabels=Class_labels,
                  yticklabels=Class_labels,
                  annot=True,
                  fmt="d")
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            cm_figuresavedpath = os.path.join(checkpoint_path,'Confusion_matrix_' + species + '_.png')
            plt.savefig(cm_figuresavedpath)

        show_confusion_matrix(test_target_h, test_predict_h, Class_labels_h, 'h')
        show_confusion_matrix(test_target_s, test_predict_s, Class_labels_s, 's')
        show_confusion_matrix(test_target_c, test_predict_c, Class_labels_c, 'c')
    
    if args.gpu:
        print('GPU INFO.....', file=f)
        print(torch.cuda.memory_summary(), end='', file=f)
