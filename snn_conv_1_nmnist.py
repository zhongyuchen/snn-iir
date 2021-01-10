import argparse
import pandas as pd
import os
import time
import sys

import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms, utils

from snn_lib.snn_layers import *
from snn_lib.optimizers import *
from snn_lib.schedulers import *
from snn_lib.data_loaders import *
import snn_lib.utilities

import omegaconf
from omegaconf import OmegaConf

if torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')

# arg parser
parser = argparse.ArgumentParser(description='conv snn')
parser.add_argument('--config_file', type=str, default='snn_conv_1_mnist.yaml',
                    help='path to configuration file')
parser.add_argument('--train', action='store_true',
                    help='train model')

parser.add_argument('--test', action='store_true',
                    help='test model')

args = parser.parse_args()

# %% config file
if args.config_file is None:
    print('No config file provided, use default config file')
else:
    print('Config file provided:', args.config_file)

conf = OmegaConf.load(args.config_file)

torch.manual_seed(conf['pytorch_seed'])
np.random.seed(conf['pytorch_seed'])

experiment_name = conf['experiment_name']

# %% checkpoint
save_checkpoint = conf['save_checkpoint']
checkpoint_base_name = conf['checkpoint_base_name']
checkpoint_base_path = conf['checkpoint_base_path']
test_checkpoint_path = conf['test_checkpoint_path']

# %% training parameters
hyperparam_conf = conf['hyperparameters']
length = hyperparam_conf['length']
batch_size = hyperparam_conf['batch_size']
synapse_type = hyperparam_conf['synapse_type']
epoch = hyperparam_conf['epoch']
tau_m = hyperparam_conf['tau_m']
tau_s = hyperparam_conf['tau_s']
filter_tau_m = hyperparam_conf['filter_tau_m']
filter_tau_s = hyperparam_conf['filter_tau_s']

membrane_filter = hyperparam_conf['membrane_filter']

train_bias = hyperparam_conf['train_bias']
train_coefficients = hyperparam_conf['train_coefficients']

# %% mnist config
dataset_config = conf['mnist_config']
max_rate = dataset_config['max_rate']
use_transform = dataset_config['use_transform']

# %% transform config
if use_transform == True:
    rand_transform = get_rand_transform(conf['transform'])
else:
    rand_transform = None

# load mnist training dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=rand_transform)
mnist_trainset, mnist_devset = random_split(mnist_trainset, [50000, 10000], generator=torch.Generator().manual_seed(42))
# load mnist test dataset
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

# acc file name
acc_file_name = experiment_name + '_' + conf['acc_file_name']


class MNISTDataset(Dataset):
    """mnist dataset

    torchvision_mnist: dataset object
    length: number of steps of snn
    max_rate: a scale factor. MNIST pixel value is normalized to [0,1], and them multiply with this value
    flatten: return 28x28 image or a flattened 1d vector
    transform: transform
    """

    def __init__(self, torchvision_mnist, length, max_rate=1, flatten=False, transform=None):
        self.dataset = torchvision_mnist
        self.transform = transform
        self.flatten = flatten
        self.length = length
        self.max_rate = max_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.dataset[idx][0]
        if self.transform:
            img = self.transform(img)

        img = np.array(self.dataset[idx][0], dtype=np.float32) / 255.0 * self.max_rate
        shape = img.shape
        img_spike = None
        if self.flatten == True:
            img = img.reshape(-1)

        return img, self.dataset[idx][1]


# %% define model
class mysnn(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.length = length
        self.batch_size = batch_size

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter

        # 1
        self.axon1 = dual_exp_iir_layer((1, 28, 28), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.conv1 = conv2d_layer(
            h_input=28, w_input=28, in_channels=1, out_channels=32, kernel_size=3,
            stride=1, padding=1, dilation=1, step_num=length, batch_size=batch_size,
            tau_m=tau_m, train_bias=train_bias, membrane_filter=membrane_filter, input_type='axon'
        )
        # 2
        self.axon2 = dual_exp_iir_layer((32, 28, 28), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.conv2 = conv2d_layer(
            h_input=28, w_input=28, in_channels=32, out_channels=32, kernel_size=3,
            stride=1, padding=1, dilation=1, step_num=length, batch_size=batch_size,
            tau_m=tau_m, train_bias=train_bias, membrane_filter=membrane_filter, input_type='axon'
        )
        # 3
        self.axon3 = dual_exp_iir_layer((32, 28, 28), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.conv3 = conv2d_layer(
            h_input=28, w_input=28, in_channels=32, out_channels=64, kernel_size=3,
            stride=1, padding=1, dilation=1, step_num=length, batch_size=batch_size,
            tau_m=tau_m, train_bias=train_bias, membrane_filter=membrane_filter, input_type='axon'
        )
        # 4
        self.axon4 = dual_exp_iir_layer((64, 28, 28), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.pool4 = maxpooling2d_layer(
            h_input=28, w_input=28, in_channels=64, kernel_size=2,
            stride=2, padding=0, dilation=1, step_num=length, batch_size=batch_size
        )
        # 5
        self.axon5 = dual_exp_iir_layer((64, 14, 14), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.conv5 = conv2d_layer(
            h_input=14, w_input=14, in_channels=64, out_channels=64, kernel_size=3,
            stride=1, padding=1, dilation=1, step_num=length, batch_size=batch_size,
            tau_m=tau_m, train_bias=train_bias, membrane_filter=membrane_filter, input_type='axon'
        )
        # 6
        self.axon6 = dual_exp_iir_layer((64, 14, 14), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.pool6 = maxpooling2d_layer(
            h_input=14, w_input=14, in_channels=64, kernel_size=2,
            stride=2, padding=0, dilation=1, step_num=length, batch_size=batch_size
        )
        # 7
        self.axon7 = dual_exp_iir_layer((7 * 7 * 64,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn7 = neuron_layer(7 * 7 * 64, 512, self.length, self.batch_size, tau_m, self.train_bias,
                                 self.membrane_filter)
        self.dropout7 = torch.nn.Dropout(p=0.3, inplace=False)
        # 8
        self.axon8 = dual_exp_iir_layer((512,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn8 = neuron_layer(512, 10, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size(784), t] -> [batch, 1, 28, 28, t]
        :return:
        """
        inputs = inputs.view(-1, 28, 28, inputs.shape[2])
        inputs = inputs.unsqueeze(1)

        # 1
        axon1_states = self.axon1.create_init_states()
        conv1_states = self.conv1.create_init_states()
        axon1_out, axon1_states = self.axon1(inputs, axon1_states)
        spike_l1, conv1_states = self.conv1(axon1_out, conv1_states)
        # 2
        axon2_states = self.axon2.create_init_states()
        conv2_states = self.conv2.create_init_states()
        axon2_out, axon2_states = self.axon2(spike_l1, axon2_states)
        spike_l2, conv2_states = self.conv2(axon2_out, conv2_states)
        # 3
        axon3_states = self.axon3.create_init_states()
        conv3_states = self.conv3.create_init_states()
        axon3_out, axon3_states = self.axon3(spike_l2, axon3_states)
        spike_l3, conv3_states = self.conv3(axon3_out, conv3_states)
        # 4
        axon4_states = self.axon4.create_init_states()
        axon4_out, axon4_states = self.axon4(spike_l3, axon4_states)
        spike_l4 = self.pool4(axon4_out)
        # 5
        axon5_states = self.axon5.create_init_states()
        conv5_states = self.conv5.create_init_states()
        axon5_out, axon5_states = self.axon5(spike_l4, axon5_states)
        spike_l5, conv5_states = self.conv5(axon5_out, conv5_states)
        # 6
        axon6_states = self.axon6.create_init_states()
        axon6_out, axon6_states = self.axon6(spike_l5, axon6_states)
        spike_l6 = self.pool6(axon6_out)
        # 6 -> 7
        spike_l6 = spike_l6.view(spike_l6.shape[0], -1, spike_l6.shape[-1])
        # 7
        axon7_states = self.axon7.create_init_states()
        snn7_states = self.snn7.create_init_states()
        axon7_out, axon7_states = self.axon7(spike_l6, axon7_states)
        spike_l7, snn7_states = self.snn7(axon7_out, snn7_states)
        drop_7 = self.dropout7(spike_l7)
        # 8
        axon8_states = self.axon8.create_init_states()
        snn8_states = self.snn8.create_init_states()
        axon8_out, axon8_states = self.axon8(drop_7, axon8_states)
        spike_l8, snn8_states = self.snn8(axon8_out, snn8_states)

        return spike_l8


########################### train function ###################################
def train(model, optimizer, scheduler, train_data_loader, writer=None):
    eval_image_number = 0
    correct_total = 0
    wrong_total = 0

    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for i_batch, sample_batched in enumerate(train_data_loader):

        x_train = sample_batched[0]
        target = sample_batched[1].to(device)
        x_train = x_train.repeat(length, 1, 1).permute(1, 2, 0).to(device)
        out_spike = model(x_train)

        spike_count = torch.sum(out_spike, dim=2)

        model.zero_grad()
        loss = criterion(spike_count, target.long())
        loss.backward()
        optimizer.step()

        # calculate acc
        _, idx = torch.max(spike_count, dim=1)

        eval_image_number += len(sample_batched[1])
        wrong = len(torch.where(idx != target)[0])

        correct = len(sample_batched[1]) - wrong
        wrong_total += len(torch.where(idx != target)[0])
        correct_total += correct
        acc = correct_total / eval_image_number

        # scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            scheduler.step()

    # scheduler step
    if isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR):
        scheduler.step()

    acc = correct_total / eval_image_number

    return acc, loss


def test(model, test_data_loader, writer=None):
    eval_image_number = 0
    correct_total = 0
    wrong_total = 0

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    for i_batch, sample_batched in enumerate(test_data_loader):
        x_test = sample_batched[0]
        target = sample_batched[1].to(device)
        x_test = x_test.repeat(length, 1, 1).permute(1, 2, 0).to(device)
        out_spike = model(x_test)

        spike_count = torch.sum(out_spike, dim=2)

        loss = criterion(spike_count, target.long())

        # calculate acc
        _, idx = torch.max(spike_count, dim=1)

        eval_image_number += len(sample_batched[1])
        wrong = len(torch.where(idx != target)[0])

        correct = len(sample_batched[1]) - wrong
        wrong_total += len(torch.where(idx != target)[0])
        correct_total += correct
        acc = correct_total / eval_image_number

    acc = correct_total / eval_image_number

    return acc, loss


if __name__ == "__main__":

    snn = mysnn().to(device)

    writer = SummaryWriter()

    params = list(snn.parameters())

    optimizer = get_optimizer(params, conf)

    scheduler = get_scheduler(optimizer, conf)

    train_data = MNISTDataset(mnist_trainset, max_rate=1, length=length, flatten=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)

    dev_data = MNISTDataset(mnist_devset, max_rate=1, length=length, flatten=True)
    dev_dataloader = DataLoader(dev_data, batch_size=batch_size, shuffle=True, drop_last=False)

    test_data = MNISTDataset(mnist_testset, max_rate=1, length=length, flatten=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)

    train_acc_list = []
    test_acc_list = []
    checkpoint_list = []

    if args.train == True:
        train_it = 0
        test_it = 0
        for j in range(epoch):

            epoch_time_stamp = time.strftime("%Y%m%d-%H%M%S")

            snn.train()
            train_acc, train_loss = train(snn, optimizer, scheduler, train_dataloader, writer=None)
            train_acc_list.append(train_acc)

            print('Train epoch: {}, acc: {}'.format(j, train_acc))

            # save every checkpoint
            if save_checkpoint == True:
                checkpoint_name = checkpoint_base_name + experiment_name + '_' + str(j) + '_' + epoch_time_stamp
                checkpoint_path = os.path.join(checkpoint_base_path, checkpoint_name)
                checkpoint_list.append(checkpoint_path)

                torch.save({
                    'epoch': j,
                    'snn_state_dict': snn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, checkpoint_path)

            # test model
            snn.eval()
            test_acc, test_loss = test(snn, dev_dataloader, writer=None)

            print('Test epoch: {}, acc: {}'.format(j, test_acc))
            test_acc_list.append(test_acc)

        # save result and get best epoch
        train_acc_list = np.array(train_acc_list)
        test_acc_list = np.array(test_acc_list)

        acc_df = pd.DataFrame(data={'train_acc': train_acc_list, 'test_acc': test_acc_list})

        acc_df.to_csv(acc_file_name)

        best_train_acc = np.max(train_acc_list)
        best_train_epoch = np.argmax(test_acc_list)

        best_test_epoch = np.argmax(test_acc_list)
        best_test_acc = np.max(test_acc_list)

        best_checkpoint = checkpoint_list[best_test_epoch]

        print('Summary:')
        print('Best train acc: {}, epoch: {}'.format(best_train_acc, best_train_epoch))
        print('Best test acc: {}, epoch: {}'.format(best_test_acc, best_test_epoch))
        print('best checkpoint:', best_checkpoint)

    elif args.test == True:

        test_checkpoint = torch.load(test_checkpoint_path)
        snn.load_state_dict(test_checkpoint["snn_state_dict"])

        test_acc, test_loss = test(snn, test_dataloader)

        print('Test checkpoint: {}, acc: {}'.format(test_checkpoint_path, test_acc))

