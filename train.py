import importlib
import os
import argparse
import time
import copy

import torch
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

import datasets as ds
import models
import utils


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Runner for PINN')
parser.add_argument('--batch_num', type=int, default=10, metavar='N',
                    help='input number of batches to be processed')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--dataset', type=str, default='',
                    help='Select Target Dataset')
parser.add_argument('--display', action='store_true')
parser.add_argument('--single_run', action='store_true')
args = parser.parse_args()
display = args.display

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

result_counter = 1
result_folder = './results/'
os.makedirs(result_folder, exist_ok=True)
dataset_result_folder = result_folder+args.dataset+'/'
os.makedirs(dataset_result_folder, exist_ok=True)

base_path = dataset_result_folder+args.dataset+'_results*#*.csv'
path = base_path.replace('*#*', str(result_counter))

while os.path.exists(path):
    result_counter += 1
    path = base_path.replace('*#*', str(result_counter))

df = pd.DataFrame(columns=['model', 'epochs', 'training_set_size', 'max_accuracy', 'min_loss', 'learning_rate'])

if args.dataset == 'AffNIST':
    from models.AffNIST import PINN, STN, CNN
    train_loader, test_loader, dataset_details = ds.AffNIST()
    batch_size, channel_num, height, width = dataset_details
    num_classes = 10

    cnn = CNN(channel_num, (width, height), num_classes=num_classes).to(device)
    pinn = PINN(channel_num, (width, height), num_classes=num_classes).to(device)
    stn = STN(channel_num, (width, height), num_classes=num_classes).to(device)

    criterion_cnn = nn.CrossEntropyLoss()
    criterion_pinn = nn.CrossEntropyLoss()
    criterion_stn = nn.CrossEntropyLoss()

elif args.dataset == 'Cifar10':
    train_loader, test_loader, dataset_details = ds.Cifar10()
    batch_size, channel_num, height, width = dataset_details
    criterion_cnn = nn.CrossEntropyLoss()
    criterion_pinn = nn.CrossEntropyLoss()
    criterion_stn = nn.CrossEntropyLoss()

elif args.dataset == "birds":
    from models.birds import PINNResnet, STNResnet
    train_loader, test_loader, dataset_details = ds.birds()
    batch_size, channel_num, height, width = dataset_details
    num_classes = 200

    pinn = PINNResnet(num_classes=200).to(device)
    stn = STNResnet(num_classes=200).to(device)
    criterion_cnn = nn.CrossEntropyLoss()
    criterion_pinn = nn.CrossEntropyLoss()
    criterion_stn = nn.CrossEntropyLoss()

elif args.dataset == 'smallNORBsingle':
    from models.smallNorb import PINNResnet, STNResnet, BasicResnet, PINN, STNBasic, CNNBasic
    train_loader, test_loader, dataset_details = ds.smallNORBsingle()
    batch_size, channel_num, height, width = dataset_details
    num_classes = 5

    cnn = CNNBasic(num_classes=num_classes).to(device)
    pinn = PINN().to(device)
    stn = STNBasic(num_classes=num_classes).to(device)
    criterion_cnn = nn.CrossEntropyLoss()
    criterion_pinn = nn.CrossEntropyLoss()
    criterion_stn = nn.CrossEntropyLoss()

elif args.dataset == 'headPose':
    from models.headPose import PINNBasic, STNBasic, CNNBasic, MultiPINNBasic
    train_loader, test_loader, dataset_details = ds.headPose()
    batch_size, channel_num, height, width = dataset_details

    cnn = CNNBasic().to(device)
    pinn = MultiPINNBasic().to(device)
    stn = STNBasic().to(device)
    criterion_cnn = nn.MSELoss()
    criterion_pinn = nn.MSELoss()
    criterion_stn = nn.MSELoss()

elif args.dataset == "svhn":
    from models.svhn import CNN, PINN, STN
    train_loader, test_loader, dataset_details = ds.svhn()
    batch_size, channel_num, height, width = dataset_details
    num_classes = 10

    cnn = CNN(num_classes).to(device)
    pinn = PINN(num_classes).to(device)
    stn = STN(num_classes).to(device)
    criterion_cnn = nn.CrossEntropyLoss()
    criterion_pinn = nn.CrossEntropyLoss()
    criterion_stn = nn.CrossEntropyLoss()

elif args.dataset == "facialRecognition":
    from models.facialRecognition import PINNBasic, CNNBasic, STNBasic
    train_loader, test_loader, dataset_details = ds.facialRecognition()
    batch_size, channel_num, height, width = dataset_details

    cnn = CNNBasic().to(device)
    pinn = PINNBasic().to(device)
    stn = STNBasic().to(device)
    criterion_cnn = nn.CrossEntropyLoss()
    criterion_pinn = nn.CrossEntropyLoss()
    criterion_stn = nn.CrossEntropyLoss()
else:
    raise ValueError("unknown dataset")

epoch_num = args.epochs
batch_num = args.batch_num
learning_rate = args.lr

cnn_loss, cnn_acc, cnn_loss_validation = ([0], [0], [0])
pinn_loss, pinn_acc, pinn_loss_validation = ([0], [0], [0])
stn_loss, stn_acc, stn_loss_validation = ([0], [0], [0])


train_cnn = True
train_pinn = True
train_stn = True

if args.single_run:
    batches = [batch_num]
else:
    batches = [5, 10, 20, 30, 40, 50, 100]
    # batches = [1, 2]

for batch_num in batches:
    training_set_size = batch_num * train_loader.batch_size
    if train_cnn:
        print("Starting CNN")
        t1 = time.time()
        cnn_copy = copy.deepcopy(cnn)
        optimizer_cnn = torch.optim.Adam(cnn_copy.parameters(), lr=learning_rate)
        _, cnn_loss, cnn_acc, cnn_loss_validation = utils.train_net(cnn_copy, train_loader, test_loader, criterion_cnn, optimizer_cnn, batch_num, epoch_num)
        print("DONE TRAINING CNN: {}s".format(time.time() - t1))
        print("FINAL ACC:", cnn_acc[-1], '\n\n')

    if train_pinn:
        print("Starting PINN")
        t2 = time.time()
        pinn_copy = copy.deepcopy(pinn)
        optimizer_pinn = torch.optim.Adam(pinn_copy.parameters(), lr=learning_rate)
        after_pinn, pinn_loss, pinn_acc, pinn_loss_validation = utils.train_net(pinn_copy, train_loader, 
                      test_loader, criterion_pinn, optimizer_pinn, batch_num, epoch_num, validation_split=0.05)
        print("DONE TRAINING PINN: {}s".format(time.time() - t2))
        print("FINAL ACC:", pinn_acc[-1], '\n\n')
        


        # rand_batch = np.random.randint(50)
        # for i, (img_batch, label_batch) in enumerate(test_loader):
        #     if i == rand_batch:
        #         break

        # utils.plot_batch(img_batch, label_batch, after_pinn, num_examples=16)

    if train_stn:
        print("Starting STN")
        t2 = time.time()
        stn_copy = copy.deepcopy(stn)
        optimizer_stn = torch.optim.Adam(stn_copy.parameters(), lr=learning_rate)
        _, stn_loss, stn_acc, stn_loss_validation = utils.train_net(stn_copy, train_loader, test_loader, criterion_stn, optimizer_stn, batch_num, epoch_num)
        print("DONE TRAINING STN: {}s".format(time.time() - t2))
        print("FINAL ACC:", stn_acc[-1], '\n\n')
   
    for model_name, loss, acc, validation_loss in zip(['cnn', 'pinn', 'stn'],
                                            [cnn_loss, pinn_loss, stn_loss],
                                            [cnn_acc, pinn_acc, stn_acc],
                                            [cnn_loss_validation, pinn_loss_validation, stn_loss_validation]):
        
        print(np.round(loss, 5))
        best_epoch = np.argmin(validation_loss)
        max_accuracy = acc[best_epoch]
        print(best_epoch, max_accuracy)
        min_loss = loss[best_epoch]
        df = df.append(pd.DataFrame([[model_name, epoch_num, training_set_size, max_accuracy, 
                        min_loss, learning_rate]], columns=df.columns), ignore_index=True)

df.to_csv(path)
training_sizes = df['training_set_size'].unique()
fig, ax = plt.subplots(1, 2, figsize=(25, 10))
fig2, ax2 = plt.subplots(1, 2, figsize=(25, 10))

for model_name in ['cnn', 'pinn', 'stn']:
    df_model = df[df['model'] == model_name]
    plot_acc = []
    plot_loss = []
    plot_values_x = []
    
    for training_size in training_sizes:
        plot_values_x.append(training_size)

        acc_values = df_model[df_model['training_set_size'] == training_size]['max_accuracy'].values
        mean_acc = acc_values.mean()
        std_acc = np.std(acc_values)
        plot_acc.append(mean_acc)

        loss_values = df_model[df_model['training_set_size'] == training_size]['min_loss'].values
        mean_loss = loss_values.mean()
        std_loss = np.std(loss_values)
        plot_loss.append(mean_loss)

    ax[0].plot(plot_values_x, plot_acc, 'o--', label=model_name)
    ax[0].errorbar(plot_values_x, plot_acc, yerr=std_acc)

    ax[1].plot(plot_values_x, plot_acc, 'o--', label=model_name)
    ax[1].errorbar(plot_values_x, plot_loss, yerr=std_loss)

    ax2[0].plot(plot_values_x, plot_acc, 'o--', label=model_name)

    ax2[1].plot(plot_values_x, plot_acc, 'o--', label=model_name)


ax[0].set_xlabel("Training Set Size")
ax[0].set_ylabel("Test Accuracy")
ax[0].legend()

ax[1].set_xlabel("Training Set Size")
ax[1].set_ylabel("Test Loss")
ax[1].legend()

ax[0].set_xlabel("Training Set Size")
ax[0].set_ylabel("Test Accuracy")
ax[0].legend()

ax[1].set_xlabel("Training Set Size")
ax[1].set_ylabel("Test Loss")
ax[1].legend()

fig.savefig(path.replace('.csv', '.png'))
fig2.savefig(path.replace('.csv', '_noerr.png'))
