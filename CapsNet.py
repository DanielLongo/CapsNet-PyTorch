import torch
import torchvision.datasets
from torchvision import transforms
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from CapsuleLayers import DenseCapsule, PrimaryCapsule

class CapsNet(nn.Module):
	def __init__(self):
		super(CapsNet, self).__init__()

		self.primary_capsule = PrimaryCapsule()
		self.digit_capsule = DenseCapsule()

		self.decoder = nn.Sequential(
			nn.Linear(16 * 10, 512),
			nn.ReLU(),
			nn.Linear(512, 1024),
			nn.ReLU(),
			nn.Linear(1024, 28 * 28),
			nn.Sigmoid())

	def forward(self, x, y=None):
		out = self.primary_capsule(x)
		out = self.digit_capsule(out)
		length = out.norm(dim=-1)
		if y is None:
			print("Y IS NONENONONONONON")
			index = length.max(dim=1)[1]
			y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.))
			print("yYYYYYYYYYY", y.shape)
		reconstruction = self.decoder((out * y[:, :, None]).view(out.size(0), -1))
		return length, reconstruction

if __name__ == "__main__":
	caps_net = CapsNet()
	mnist_train = torchvision.datasets.MNIST('./MNIST_data', train=True, download=True, transform=transforms.ToTensor())
	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=16, shuffle=True)
	for x, y in train_loader:
		X = x
		Y = y
		break
	Y = torch.zeros(Y.size(0), 10).scatter_(1, Y.view(-1, 1), 1.)
	print(X.shape, Y.shape)
	a,b = caps_net(x, y=Variable(y))
	print("a", a.shape)
	print("b", b.shape)
	print("finished")

