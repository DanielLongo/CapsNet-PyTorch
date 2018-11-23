'''
Pytorch Implementation of "Dynamic Routing Between Capsules"
	input: MNIST (n, 28, 28, 1)
	Primary capsules -> (n, 1152, 8)

'''
import torchvision.datasets
from torchvision import transforms
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


def squash(x, dim=-1):
	squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
	scale = squared_norm / (1 + squared_norm)
	out = scale * x / torch.sqrt(squared_norm)
	return out

class PrimaryCapsule(nn.Module):
	def __init__(self):		
		super(PrimaryCapsule, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 256, kernel_size=9, stride=1, padding=0),
			nn.ReLU()
		)
		
		self.conv2 = nn.Sequential(
			nn.Conv2d(256, caps1_n_maps * caps1_n_dims, kernel_size=9, stride=2, padding=0),
			nn.ReLU()
		)

	def forward(self, x):
		# 28, 28, 1
		print("x", x.shape)
		out = self.conv1(x)
		print("out", out.shape)
		# 20, 20, 256
		out = self.conv2(out)
		print("out", out.shape)
		# 6, 6, 256
		out = out.view(-1, caps1_n_caps, caps1_n_dims)
		print("out", out.shape)
		out = squash(out)
		print("out", out.shape)
		return out

caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6 #1152 primary capsules
caps1_n_dims = 8

class DigitCapsule(nn.Module):
	def __init__(self):		
		super(DigitCapsule, self).__init__()
		self.weight = nn.Parameter(.01 * torch.randn(caps2_n_caps, caps1_n_caps, caps2_n_dims, caps1_n_dims))
		self.routings = 3 
	def forward(self, x):
		print("x", x.shape)
		x = x[:, None, :, :, None] #expands dims
		print("x", x.shape)
		print("w", self.weight.shape)
		x_hat = torch.squeeze(torch.matmul(self.weight, x), dim=-1)

		x_hat_detached = x_hat.detach()
		print(x_hat_detached.shape)

		b = Variable(torch.zeros(x.shape[0], caps2_n_caps, caps1_n_caps))

		assert self.routings > 0
		for i in range(self.routings):
			c = F.softmax(b, dim=1)
			if i == self.routings - 1:
				out = squash(torch.sum(c[:,:,:, None] * x_hat, dim=-2, keepdim=True))
			else: #no gradeinets here
				outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
				b = b + torch.sum(outputs * x_hat_detached, dim =-1)
		return torch.squeeze(outputs, dim=-2)



caps2_n_caps = 10
caps2_n_dims = 16

mnist_train = torchvision.datasets.MNIST('./MNIST_data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=16, shuffle=True)
for x, y in train_loader:
	X = x
	Y = y
	break
print(X.shape, Y.shape)
primary = PrimaryCapsule()
digit = DigitCapsule()
z = primary(X)
digit(z)
print("finished")
