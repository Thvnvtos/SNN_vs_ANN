import torch, json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import models

config_file_path = "config.json"

with open(config_file_path) as f:
		config = json.load(f)

data_root = config["data_root"]
seed = config["seed"]
batch_size = config["batch_size"]

log_interval = 60000//(4*batch_size) # in order to have 5 logs in each epoch depending on the batch_size
mnist_mean = 0.1307
mnist_std = 0.3081


def train_full(net, train_loader, optimizer, device, epoch):
	
	net.train()
	correct_pred = 0

	for batch_idx, (x, label) in enumerate(train_loader):
		x, label = x.to(device), label.to(device)
		optimizer.zero_grad()
		y = net(x)
		loss = F.cross_entropy(y, label)
		pred = y.argmax(dim=1)
		correct_pred += (pred == label).sum().item()
		loss.backward()
		optimizer.step()

		if batch_idx % log_interval == 0:
			print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(x), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

	print("===> Train Epoch Accuracy : {:.2f}%".format(100. * correct_pred / len(train_loader.dataset)))

def test_full(net, test_loader, device):
	net.eval()
	test_loss = 0
	correct_pred = 0
	with torch.no_grad():
		for x, label in test_loader:
			x, label = x.to(device), label.to(device)
			y = net(x)
			test_loss += F.cross_entropy(y, label).item()
			pred = y.argmax(dim=1)
			correct_pred += (pred == label).sum().item()

	print("===> Test Accuracy : {:.2f}%, Test Average loss: {:.4f}".format(100. * correct_pred / len(test_loader.dataset), 
		test_loss / len(test_loader.dataset)))

if __name__ == '__main__':

	torch.manual_seed(seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		print("Training on {} !".format(torch.cuda.get_device_name()))
	else:
		print("Training on CPU :( ")

	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((mnist_mean,), (mnist_std,))
		])

	dataset_train = datasets.MNIST(data_root, train=True, transform=transform)
	dataset_test = datasets.MNIST(data_root, train=False,transform=transform)

	train_loader = torch.utils.data.DataLoader(dataset_train, batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset_test, batch_size)


	if config["train_ann"]:
		config_ann = config["ann"]

		net = models.ANN().to(device)
		optimizer = optim.SGD(net.parameters(), lr = config_ann["lr"])

		epochs = config_ann["epochs"]
		print("########## Training ANN for {} Epochs ##########\n".format(epochs))
		for epoch in range(epochs):
			train_full(net, train_loader, optimizer, device, epoch+1)
			test_full(net, test_loader, device)
			print("------------------------------------------------------")