import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class CapsuleLayer(nn.Module):
	def __init__(self, channels_in, channels_out, num_capsules, num_route_nodes, kernel_size=None, stride=None, num_iterations=3):		
		super(CapsuleLayer, self).__init__()
		self.num_route_nodes = num_route_nodes
		self.num_iterations = num_iterations
		self.num_capsules = num_capsules
		


class CapsNet(nn.Module):
	def __init__(self):
		super(CapsNet, self).__init__()
		self.conv1 = nn.Con2d(1, 256, kernel_size=[9,9], stride=[1,1])
