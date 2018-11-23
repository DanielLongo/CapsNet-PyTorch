'''
Pytorch Implementation of "Dynamic Routing Between Capsules"
	input: MNIST (n, 28, 28, 1) ->
	Primary Capsules (n, 6, 6, 8) ->
'''
import torch
from torch import nn

def squash(self, x, dim=-1):
	squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
	scale = squared_norm / (1 + squared_norm)
	out = scale * x / torch.sqrt(squared_norm)

class PrimaryCapsule(nn.Module):
	def __init__(self):		
		super(PrimaryCapsule, self).__init__()
		self.conv1 = nn.Conv2d(1,)
		self.conv2 = nn.Conv2d()

	def forward(self, x):
		out = conv1(x)
		out = conv2(out)
		out.view(-1, 6, 6)
		out = squash(out)
		return out


class DigitCapsules(nn.Module):
	def __init__(self):
		super(DigitCapsules, self).__init__()
		self.w = nn.Linear(shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims))

	def forward(self, x):
		batch_size = x.shape[0]
		


