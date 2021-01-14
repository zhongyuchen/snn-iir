import argparse
import pandas as pd
import os
import time
import sys
import csv
import struct

import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import utils

from snn_lib.snn_layers import *
from snn_lib.optimizers import *
from snn_lib.schedulers import *
from snn_lib.data_loaders import *
import snn_lib.utilities

import omegaconf
from omegaconf import OmegaConf

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# arg parser
parser = argparse.ArgumentParser(description='conv snn')
parser.add_argument('--config_file', type=str, default='snn_conv_1_gesture.yaml',
                    help='path to configuration file')
parser.add_argument('--train', action='store_true',
                    help='train model')

parser.add_argument('--test', action='store_true',
                    help='test model')
parser.add_argument('--load', action='store_true', help='load dataloader')

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

# acc file name
acc_file_name = experiment_name + '_' + conf['acc_file_name']


class GestureDataset(Dataset):
    def __init__(self, root, train, length):
        super(GestureDataset, self).__init__()
        self.root = root
        if train is True:
            self.trial_path = 'trials_to_train.txt'
        else:
            self.trial_path = 'trials_to_test.txt'
        self.length = length
        self.data, self.label = self.get_dataset()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def get_event(self, file):
        with open(os.path.join(self.root, file), 'rb') as bin_f:
            # skip ascii header
            line = bin_f.readline()
            while line.startswith(b'#'):
                if line == b'#!END-HEADER\r\n':
                    break
                else:
                    line = bin_f.readline()

            x_list, y_list, t_list, p_list = [], [], [], []
            while True:
                header = bin_f.read(28)
                if not header or len(header) == 0:
                    break

                # read header
                e_type = struct.unpack('H', header[0:2])[0]
                e_source = struct.unpack('H', header[2:4])[0]
                e_size = struct.unpack('I', header[4:8])[0]
                e_offset = struct.unpack('I', header[8:12])[0]
                e_tsoverflow = struct.unpack('I', header[12:16])[0]
                e_capacity = struct.unpack('I', header[16:20])[0]
                e_number = struct.unpack('I', header[20:24])[0]
                e_valid = struct.unpack('I', header[24:28])[0]

                data_length = e_capacity * e_size
                data = bin_f.read(data_length)
                counter = 0

                if e_type == 1:
                    while data[counter:counter + e_size]:
                        aer_data = struct.unpack('I', data[counter:counter + 4])[0]
                        timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0] | e_tsoverflow << 31
                        x = (aer_data >> 17) & 0x00007FFF
                        y = (aer_data >> 2) & 0x00007FFF
                        pol = (aer_data >> 1) & 0x00000001
                        counter = counter + e_size
                        x_list.append(x)
                        y_list.append(y)
                        t_list.append(timestamp)
                        p_list.append(pol)
                else:
                    # non-polarity event packet, not implemented
                    pass
            p_list = torch.tensor(p_list, dtype=torch.int64)
            x_list = torch.tensor(x_list, dtype=torch.int64)
            y_list = torch.tensor(y_list, dtype=torch.int64)
            t_list = torch.tensor(t_list, dtype=torch.int64)
            return p_list, x_list, y_list, t_list

    def get_label(self, file):
        data = []
        with open(os.path.join(self.root, file), 'r') as f:
            lines = csv.reader(f)
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                data.append([int(d) for d in line])
        return data

    def get_single_sample(self, event):
        p, x, y, t = event
        t = (t - t.min()) // 1000
        bin_width = max((t.max() + 1), 300) // self.length
        t = t // bin_width
        spike_train = torch.zeros((2, 128, 128, max(self.length, t.max() + 1)), dtype=torch.bool)  # [p, x, y, t]
        spike_train[p, x, y, t] = True
        spike_train = spike_train[:, :, :, 0:self.length]
        return spike_train

    def get_sample(self, file):
        print('process:', file)
        p, x, y, t = self.get_event(file=file)
        period = self.get_label(file=file.split('.')[0] + '_labels.csv')
        data, label = [], []
        for peri in period:
            index = (t >= peri[1]) * (t <= peri[2])
            index = index.long()
            data.append(self.get_single_sample(event=(p[index], x[index], y[index], t[index])))
            label.append(peri[0] - 1)
        return torch.stack(data), torch.tensor(label)  # [batch, p, x, y, t]

    def get_trial(self):
        with open(os.path.join(self.root, self.trial_path), 'r') as f:
            file_list = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    file_list.append(line)
            return file_list

    def get_dataset(self):
        data = []
        label = []
        for file in self.get_trial():
            x, y = self.get_sample(file=file)
            data.append(x)
            label.append(y)
        return torch.cat(data), torch.cat(label)


# %% define model
class mysnn(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.length = length
        self.batch_size = batch_size

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter

        # 1: 2x128x128 -> 64x42x42
        self.axon1 = dual_exp_iir_layer((2, 128, 128), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.conv1 = conv2d_layer(
            h_input=128, w_input=128, in_channels=2, out_channels=64, kernel_size=7,
            stride=3, padding=1, dilation=1, step_num=length, batch_size=batch_size,
            tau_m=tau_m, train_bias=train_bias, membrane_filter=membrane_filter, input_type='axon'
        )
        # 2: 64x42x42 -> 32x14x14
        self.axon2 = dual_exp_iir_layer((64, 42, 42), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.conv2 = conv2d_layer(
            h_input=42, w_input=42, in_channels=64, out_channels=32, kernel_size=3,
            stride=3, padding=0, dilation=1, step_num=length, batch_size=batch_size,
            tau_m=tau_m, train_bias=train_bias, membrane_filter=membrane_filter, input_type='axon'
        )
        # 3: 32x14x14-> 32x14x14
        self.axon3 = dual_exp_iir_layer((32, 14, 14), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.conv3 = conv2d_layer(
            h_input=14, w_input=14, in_channels=32, out_channels=32, kernel_size=3,
            stride=1, padding=1, dilation=1, step_num=length, batch_size=batch_size,
            tau_m=tau_m, train_bias=train_bias, membrane_filter=membrane_filter, input_type='axon'
        )
        # 4
        self.axon4 = dual_exp_iir_layer((32 * 14 * 14,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn4 = neuron_layer(32 * 14 * 14, 256, self.length, self.batch_size, tau_m, self.train_bias,
                                 self.membrane_filter)
        self.dropout4 = torch.nn.Dropout(p=0.3, inplace=False)
        # 5
        self.axon5 = dual_exp_iir_layer((256,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn5 = neuron_layer(256, 10, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

    def forward(self, inputs):
        """
        :param inputs: [batch, 2, 34, 34, t]
        :return:
        """
        # 1
        axon1_out, _ = self.axon1(inputs, self.axon1.create_init_states())
        spike_l1, _ = self.conv1(axon1_out,self.conv1.create_init_states())
        # 2
        axon2_out, _ = self.axon2(spike_l1, self.axon2.create_init_states())
        spike_l2, _ = self.conv2(axon2_out, self.conv2.create_init_states())
        # 3
        axon3_out, _ = self.axon3(spike_l2, self.axon3.create_init_states())
        spike_l3, _ = self.conv3(axon3_out, self.conv3.create_init_states())
        # 3 -> 4
        spike_l3 = spike_l3.view(spike_l3.shape[0], -1, spike_l3.shape[-1])
        # 4
        axon4_out, _ = self.axon4(spike_l3, self.axon4.create_init_states())
        spike_l4, _ = self.snn4(axon4_out, self.snn4.create_init_states())
        drop_4 = self.dropout4(spike_l4)
        # 5
        axon5_out, _ = self.axon5(drop_4, self.axon5.create_init_states())
        spike_l5, _ = self.snn5(axon5_out, self.snn5.create_init_states())

        return spike_l5


########################### train function ###################################
def train(model, optimizer, scheduler, train_data_loader, writer=None):
    eval_image_number = 0
    correct_total = 0
    wrong_total = 0

    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for i_batch, sample_batched in enumerate(train_data_loader):

        x_train = sample_batched[0].to(device)
        target = sample_batched[1].to(device)
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
        x_test = sample_batched[0].to(device)
        target = sample_batched[1].to(device)
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

    train_acc_list = []
    test_acc_list = []
    checkpoint_list = []

    if args.train == True:
        if args.load is True:
            print('load data')
            train_data = torch.load('./data/DvsGesture/train_data.pt')
        else:
            print('process data')
            train_data = GestureDataset(root='./data/DvsGesture', train=True, length=length)
            torch.save(train_data, './data/DvsGesture/train_data.pt')
        train_data, dev_data = random_split(
            train_data, [50000, 10000], generator=torch.Generator().manual_seed(42)
        )
        print('train_data', len(train_data), 'dev_data', len(dev_data))
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        dev_dataloader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, drop_last=True)

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
        if args.load is True:
            print('load data')
            test_data = torch.load('./data/DvsGesture/test_data.pt')
        else:
            print('process data')
            test_data = GestureDataset(root='./data/DvsGesture', train=False, length=length)
            torch.save(test_data, './data/DvsGesture/test_data.pt')
        print('test_data', len(test_data))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

        test_checkpoint = torch.load(test_checkpoint_path)
        snn.load_state_dict(test_checkpoint["snn_state_dict"])

        test_acc, test_loss = test(snn, test_dataloader)

        print('Test checkpoint: {}, acc: {}'.format(test_checkpoint_path, test_acc))

