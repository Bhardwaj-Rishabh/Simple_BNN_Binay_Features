from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import datasets, transforms

import pandas as pd
import numpy as np
from tqdm import tqdm

from module import BinaryLinear, BinaryStraightThrough

class BNN_1blk_20(nn.Module):
	def __init__(self, in_features, out_features):#num_units=4096):
		super(BNN_1blk_20, self).__init__()
		#
		self.infl_ratio=1
		self.num_internal_blocks = 1
		self.output_size = out_features
		self.input_size = in_features
		self.name = 'bnn_1blk_20'
		self.fc1 = BinaryLinear(self.input_size, 20*self.infl_ratio)
		self.bn1 = nn.BatchNorm1d(20*self.infl_ratio, eps=1e-4)
		self.htanh1 = BinaryStraightThrough()
		self.fc5 = BinaryLinear(20*self.infl_ratio, self.output_size)
		self.logsoftmax = nn.LogSoftmax()
	#
	def forward(self, x):
		#x = nn.Dropout(p=0.2)(x)
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.htanh1(x)
		#x = nn.Dropout()(x)
		x = self.fc5(x)
		return self.logsoftmax(x)

class BNN_1blk_50(nn.Module):
	def __init__(self, in_features, out_features):#num_units=4096):
		super(BNN_1blk_50, self).__init__()
		#
		self.infl_ratio=1
		self.num_internal_blocks = 1
		self.output_size = out_features
		self.input_size = in_features
		self.name = 'bnn_1blk_50'
		self.fc1 = BinaryLinear(self.input_size, 50*self.infl_ratio)
		self.bn1 = nn.BatchNorm1d(50*self.infl_ratio, eps=1e-4)
		self.htanh1 = BinaryStraightThrough()
		self.fc5 = BinaryLinear(50*self.infl_ratio, self.output_size)
		self.logsoftmax = nn.LogSoftmax()
	#
	def forward(self, x):
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.htanh1(x)
		x = self.fc5(x)
		return self.logsoftmax(x)

class BNN_1blk_100(nn.Module):
	def __init__(self, in_features, out_features):#num_units=4096):
		super(BNN_1blk_100, self).__init__()
		#
		self.infl_ratio=1
		self.num_internal_blocks = 1
		self.output_size = out_features
		self.input_size = in_features
		self.name = 'bnn_1blk_100'
		self.fc1 = BinaryLinear(self.input_size, 100*self.infl_ratio)
		self.bn1 = nn.BatchNorm1d(100*self.infl_ratio, eps=1e-4)
		self.htanh1 = BinaryStraightThrough()
		self.fc5 = BinaryLinear(100*self.infl_ratio, self.output_size)
		self.logsoftmax = nn.LogSoftmax()
	#
	def forward(self, x):
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.htanh1(x)
		x = self.fc5(x)
		return self.logsoftmax(x)


class BNN_2blk_25_10(nn.Module):
	def __init__(self, in_features, out_features):#num_units=4096):
		super(BNN_2blk_25_10, self).__init__()
		#
		self.infl_ratio=1
		self.num_internal_blocks = 1
		self.output_size = out_features
		self.input_size = in_features
		self.name = 'bnn_2blk_25_10'
		self.fc1 = BinaryLinear(self.input_size, 25*self.infl_ratio)
		self.bn1 = nn.BatchNorm1d(25*self.infl_ratio, eps=1e-4)
		self.htanh1 = BinaryStraightThrough()
		self.fc2 = BinaryLinear(25, 10*self.infl_ratio)
		self.bn2 = nn.BatchNorm1d(10*self.infl_ratio, eps=1e-4)
		self.htanh2 = BinaryStraightThrough()
		self.fc5 = BinaryLinear(10*self.infl_ratio, self.output_size)
		self.logsoftmax = nn.LogSoftmax()
	#
	def forward(self, x):
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.htanh1(x)
		x = self.fc2(x)
		x = self.bn2(x)
		x = self.htanh2(x)
		x = self.fc5(x)
		return self.logsoftmax(x)


class BNN_2blk_50_20(nn.Module):
	def __init__(self, in_features, out_features):#num_units=4096):
		super(BNN_2blk_50_20, self).__init__()
		#
		self.infl_ratio=1
		self.num_internal_blocks = 1
		self.output_size = out_features
		self.input_size = in_features
		self.name = 'bnn_2blk_50_20'
		self.fc1 = BinaryLinear(self.input_size, 50*self.infl_ratio)
		self.bn1 = nn.BatchNorm1d(50*self.infl_ratio, eps=1e-4)
		self.htanh1 = BinaryStraightThrough()
		self.fc2 = BinaryLinear(50, 20*self.infl_ratio)
		self.bn2 = nn.BatchNorm1d(20*self.infl_ratio, eps=1e-4)
		self.htanh2 = BinaryStraightThrough()
		self.fc5 = BinaryLinear(20*self.infl_ratio, self.output_size)
		self.logsoftmax = nn.LogSoftmax()
	#
	def forward(self, x):
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.htanh1(x)
		x = self.fc2(x)
		x = self.bn2(x)
		x = self.htanh2(x)
		x = self.fc5(x)
		return self.logsoftmax(x)

class BNN_2blk_100_50(nn.Module):
	def __init__(self, in_features, out_features):#num_units=4096):
		super(BNN_2blk_100_50, self).__init__()
		#
		self.infl_ratio=1
		self.num_internal_blocks = 1
		self.output_size = out_features
		self.input_size = in_features
		self.name = 'bnn_2blk_100_50'
		self.fc1 = BinaryLinear(self.input_size, 100*self.infl_ratio)
		self.bn1 = nn.BatchNorm1d(100*self.infl_ratio, eps=1e-4)
		self.htanh1 = BinaryStraightThrough()
		self.fc2 = BinaryLinear(100, 50*self.infl_ratio)
		self.bn2 = nn.BatchNorm1d(50*self.infl_ratio, eps=1e-4)
		self.htanh2 = BinaryStraightThrough()
		self.fc5 = BinaryLinear(50*self.infl_ratio, self.output_size)
		self.logsoftmax = nn.LogSoftmax()
	#
	def forward(self, x):
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.htanh1(x)
		x = self.fc2(x)
		x = self.bn2(x)
		x = self.htanh2(x)
		x = self.fc5(x)
		return self.logsoftmax(x)


class BNN_3blk_200_100(nn.Module):
	def __init__(self, in_features, out_features):#num_units=4096):
		super(BNN_3blk_200_100, self).__init__()
		#
		self.infl_ratio=1
		self.num_internal_blocks = 1
		self.output_size = out_features
		self.input_size = in_features
		self.name = 'bnn_3blk_20_100'
		self.fc1 = BinaryLinear(self.input_size, 200*self.infl_ratio)
		self.bn1 = nn.BatchNorm1d(200*self.infl_ratio, eps=1e-4)
		self.htanh1 = BinaryStraightThrough()
		self.fc2 = BinaryLinear(200, 100*self.infl_ratio)
		self.bn2 = nn.BatchNorm1d(100*self.infl_ratio, eps=1e-4)
		self.htanh2 = BinaryStraightThrough()
		self.fc3 = BinaryLinear(100, 100*self.infl_ratio)
		self.bn3 = nn.BatchNorm1d(100*self.infl_ratio, eps=1e-4)
		self.htanh3 = BinaryStraightThrough()
		self.fc5 = BinaryLinear(100*self.infl_ratio, self.output_size)
		self.logsoftmax = nn.LogSoftmax()
	#
	def forward(self, x):
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.htanh1(x)
		x = self.fc2(x)
		x = self.bn2(x)
		x = self.htanh2(x)
		x = self.fc3(x)
		x = self.bn3(x)
		x = self.htanh3(x)
		x = self.fc5(x)
		return self.logsoftmax(x)


class BNN_4blk_200_100(nn.Module):
	def __init__(self, in_features, out_features):#num_units=4096):
		super(BNN_4blk_200_100, self).__init__()
		#
		self.infl_ratio=1
		self.num_internal_blocks = 1
		self.output_size = out_features
		self.input_size = in_features
		self.name = 'bnn_4blk_20_100'
		self.fc1 = BinaryLinear(self.input_size, 200*self.infl_ratio)
		self.bn1 = nn.BatchNorm1d(200*self.infl_ratio, eps=1e-4)
		self.htanh1 = BinaryStraightThrough()
		self.fc2 = BinaryLinear(200, 100*self.infl_ratio)
		self.bn2 = nn.BatchNorm1d(100*self.infl_ratio, eps=1e-4)
		self.htanh2 = BinaryStraightThrough()
		self.fc3 = BinaryLinear(100, 100*self.infl_ratio)
		self.bn3 = nn.BatchNorm1d(100*self.infl_ratio, eps=1e-4)
		self.htanh3 = BinaryStraightThrough()
		self.fc4 = BinaryLinear(100, 100*self.infl_ratio)
		self.bn4 = nn.BatchNorm1d(100*self.infl_ratio, eps=1e-4)
		self.htanh4 = BinaryStraightThrough()
		self.fc5 = BinaryLinear(100*self.infl_ratio, self.output_size)
		self.logsoftmax = nn.LogSoftmax()
	#
	def forward(self, x):
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.htanh1(x)
		x = self.fc2(x)
		x = self.bn2(x)
		x = self.htanh2(x)
		x = self.fc3(x)
		x = self.bn3(x)
		x = self.htanh3(x)
		x = self.fc4(x)
		x = self.bn4(x)
		x = self.htanh4(x)
		x = self.fc5(x)
		return self.logsoftmax(x)