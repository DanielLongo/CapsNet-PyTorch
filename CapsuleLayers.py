"""
Some key layers used for constructing a Capsule Network. These layers can used to construct CapsNet on other dataset,
not just on MNIST.
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Pytorch`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.
    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    def forward(self, x):
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)

        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        x_hat_detached = x_hat.detach()

        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)


# class PrimaryCapsule(nn.Module):
#     """
#     Apply Conv2D with `out_channels` and then reshape to get capsules
#     :param in_channels: input channels
#     :param out_channels: output channels
#     :param dim_caps: dimension of capsule
#     :param kernel_size: kernel size
#     :return: output tensor, size=[batch, num_caps, dim_caps]
#     """
#     def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
#         super(PrimaryCapsule, self).__init__()
#         self.dim_caps = dim_caps
#         self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

#     def forward(self, x):
#         outputs = self.conv2d(x)
#         outputs = outputs.view(x.size(0), -1, self.dim_caps)
#         return squash(outputs)

# '''
# Pytorch Implementation of "Dynamic Routing Between Capsules"
# 	input: MNIST (n, 28, 28, 1)
# 	Primary capsules -> (n, 1152, 8)

# '''
# import torchvision.datasets
# from torchvision import transforms
# import torch
# from torch import nn
# from torch.autograd import Variable
# import torch.nn.functional as F


# def squash(x, dim=-1):
# 	squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
# 	scale = squared_norm / (1 + squared_norm)
# 	out = scale * x / torch.sqrt(squared_norm)
# 	return out

class PrimaryCapsule(nn.Module):
	def __init__(self):		
		super(PrimaryCapsule, self, num_maps=32, num_dims=8).__init__()
		self.num_maps = num_maps
		self.num_caps = 6 * 6 * self.num_maps
		self.num_dims = num_dims
		self.conv1 = nn.Sequential(
			nn.Conv2d(256, self.num_maps * self.num_dims, kernel_size=9, stride=2, padding=0),
			nn.ReLU()
		)

	def forward(self, x):
		# 20, 20, 256
		out = self.conv1(x)
		# 6, 6, 256
		out = out.view(-1, self.num_caps, self.num_dims)
		out = squash(out)
		return out

# caps1_n_maps = 32
# caps1_n_caps = caps1_n_maps * 6 * 6 #1152 primary capsules
# caps1_n_dims = 8

# class DenseCapsule(nn.Module):
# 	def __init__(self):		
# 		super(DenseCapsule, self).__init__()
# 		self.weight = nn.Parameter(.01 * torch.randn(caps2_n_caps, caps1_n_caps, caps2_n_dims, caps1_n_dims))
# 		self.routings = 3 
# 	def forward(self, x):
# 		print("x", x.shape)
# 		x = x[:, None, :, :, None] #expands dims
# 		print("x", x.shape)
# 		print("w", self.weight.shape)
# 		x_hat = torch.squeeze(torch.matmul(self.weight, x), dim=-1)

# 		x_hat_detached = x_hat.detach()
# 		print(x_hat_detached.shape)

# 		b = Variable(torch.zeros(x.shape[0], caps2_n_caps, caps1_n_caps))

# 		assert self.routings > 0
# 		for i in range(self.routings):
# 			c = F.softmax(b, dim=1)
# 			if i == self.routings - 1:
# 				out = squash(torch.sum(c[:,:,:, None] * x_hat, dim=-2, keepdim=True))
# 			else: #no gradeinets here
# 				outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
# 				b = b + torch.sum(outputs * x_hat_detached, dim =-1)
# 		return torch.squeeze(outputs, dim=-2)



# caps2_n_caps = 10
# caps2_n_dims = 16