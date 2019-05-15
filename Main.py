import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import datasets, transforms

from pytorchtools import EarlyStopping

import pandas as pd
import numpy as np
from tqdm import tqdm

import pickle as pk

import argparse

parser = argparse.ArgumentParser(description='Binary Neural Networks')
parser.add_argument('--binary', type=str, default='BNN',
		help='BinaryConnect or BinaryNet')
parser.add_argument('--cuda', type=bool, default=True,
		help='Use cuda or not')
parser.add_argument('--in_features', type=int, default=500,
		help='input features dim')
parser.add_argument('--out_features', type=int, default=100,
		help='output features dim')
parser.add_argument('--batch_size', type=int, default=100,
		help='batch size')
parser.add_argument('--test_batch_size', type=int, default=1000,
		help='batch size')
parser.add_argument('--lr', type=float, default=0.001,
		help='Learning rate')
parser.add_argument('--epochs', type=int, default=100,
		help='Epochs')
args = parser.parse_args()

from weight_clip import weight_clip

bndata_ = pk.load(open('./data/Binary501.pkl','rb'))
'''
convert (0 and 1) to (-1 and 1)
'''
bndata = bndata_.copy()
bndata[:,1:] -= 1
bndata[:,1:] += bndata_[:,1:]

split = int(0.8 * len(bndata))
index_list = list(range(len(bndata)))

from torch.utils.data.sampler import SubsetRandomSampler
train_idx = index_list[:split]
tr_sampler = SubsetRandomSampler(train_idx)

from BNN_Blocks import *

models = [
		  BNN_1blk_20(args.in_features, args.out_features), 
		  BNN_1blk_50(args.in_features, args.out_features), 
		  BNN_1blk_100(args.in_features, args.out_features), 
		  BNN_2blk_25_10(args.in_features, args.out_features), 
		  BNN_2blk_50_20(args.in_features, args.out_features), 
		  BNN_2blk_100_50(args.in_features, args.out_features), 
		  BNN_3blk_200_100(args.in_features, args.out_features), 
		  BNN_4blk_200_100(args.in_features, args.out_features)
		  ]

def store_data(tloss, tacc, vloss, vacc, fname):
	with open(fname, 'w') as f:
		f.write('Tr_loss\tV_loss\tTr_acc\tV_acc\n')
		for i in range(len(tloss)):
			f.write(str(round(tloss[i],3)))
			f.write('\t')
			f.write(str(round(vloss[i],3)))
			f.write('\t')
			f.write(str(round(tacc[i],3)))
			f.write('\t')
			f.write(str(round(vacc[i],3)))
			f.write('\n')

from torch.utils.data.sampler import SubsetRandomSampler

def Factory(model):
	train_losses = []
	train_acc = []
	val_losses = []
	val_acc = []
	split = int(0.8 * len(bndata))
	early_stopping = EarlyStopping(in_dim=bndata.shape[1]-1, dir_chk='chkpts', dataset='purchase', patience=10, verbose=True)
	def train(args): 
		kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
		train_loader = data.DataLoader(bndata, batch_size=1000, sampler=tr_sampler)
		all_train_loader = torch.from_numpy(bndata[:split,:])
		test_loader = torch.from_numpy(bndata[split:,:])
		net = model #BNN_1blk_100(args.in_features, args.out_features)
		print(net)
		net.cuda()
		optimizer = optim.Adam(net.parameters(), lr=args.lr)
		creterion = nn.CrossEntropyLoss()
		for epoch in range(1, args.epochs+1):
			train_epoch(epoch, net, creterion, optimizer, train_loader, args)
			tr_loss, tr_acc = test_epoch(net, creterion, all_train_loader, args)
			v_loss, v_acc = test_epoch(net, creterion, test_loader, args)
			train_losses.append(tr_loss)
			train_acc.append(tr_acc)
			val_losses.append(v_loss)
			val_acc.append(v_acc)
			early_stopping(v_loss, model, epoch)
			if early_stopping.early_stop:
				print("Early Stopping")
				store_data(train_losses, train_acc, val_losses, val_acc, 'chkpts/'+model.name+'.txt')
				break

	def train_epoch(epoch, net, creterion, optimizer, train_loader, args, valid_data=None):
		losses = 0
		accs = 0
		net.cuda()
		net.train()
		#for batch_idx, (data, target) in enumerate(tqdm(train_loader), 1):
		batch_idx = 0
		for d in train_loader:
			data = d[:, 1:].type(torch.FloatTensor).cuda()
			target = d[:,0].cuda()
			#data, target = data.cuda(), target.cuda()
			#data, target = Variable(data.view(args.batch_size, -1)), Variable(target)
		
			optimizer.zero_grad()
		
			output = net(data)
			loss = creterion(output, target.squeeze_())
			loss.backward()
			optimizer.step()
			weight_clip(net.parameters())
		
			y_pred = torch.max(output, 1)[1]
			accs += (torch.mean((y_pred == target).float())).item()#.data[0]
		
			losses += loss.item()
			batch_idx += 1
			print("Epoch {0}: Train Loss={1:.3f}, Train Accuracy={2:.3f}".format(epoch, losses / batch_idx, accs / batch_idx))
		
		if valid_data is not None:
			pass
		
	def test_epoch(net, creterion, test_loader, args):
		net.eval()
		losses = 0
		accs = 0
		#for batch_idx, (data, target) in enumerate(test_loader, 1):
		data = test_loader[:, 1:].type(torch.FloatTensor).cuda()
		target = test_loader[:,0].cuda()
		output = net(data)
		loss = creterion(output, target.squeeze_())
		
		y_pred = torch.max(output, 1)[1]
		accs += (torch.mean((y_pred == target).float())).item()
		losses += loss.item()
		print("\tTest Loss={0:.3f}, Test Accuracy={1:.3f}".format(losses, accs))
		return losses, accs
		
	train(args)

for i in range(8):
	Factory(models[i])

