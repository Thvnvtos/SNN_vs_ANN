import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from spikingjelly.clock_driven import neuron, functional, surrogate, layer


def train(net, mode, train_loader, optimizer, device, epoch, loss_f, custom_plasticity):
	
	net.train()
	correct_pred, train_loss = 0, 0

	for batch_idx, (x, label) in enumerate(train_loader):
		
		x, label = x.to(device), label.to(device)
		optimizer.zero_grad()
		y = net(x)

		if loss_f == "mse":
			label = F.one_hot(label, 10).float()
			loss = F.mse_loss(y, label)
			correct_pred += (y.argmax(dim=1) == label.argmax(dim=1)).sum().item()
		if loss_f == "ce":
			loss = F.cross_entropy(y,label)
			pred = y.argmax(dim=1)
			correct_pred += (pred == label).sum().item()

		loss.backward()
		
		if custom_plasticity:
			for p in net.parameters():
				q1 = p.quantile(0.95).item()
				q2 = (-p).quantile(0.95).item()
				p.grad = (p<q1)*(-p<q2)*p.grad
				#p.grad = (p>=q)*(p.grad/100) + (p<q)*(p.grad)

		optimizer.step()
		train_loss += loss.item()
		if mode=="snn":
			functional.reset_net(net)

	train_acc = 100. * correct_pred / len(train_loader.dataset)
	train_loss /= len(train_loader)

	print("\n===> Train Epoch Accuracy : {:.2f}%, , Train Average loss: {:.8f}".format(train_acc, train_loss))
	return train_loss, train_acc




def test(net, mode, test_loader, device, loss_f):
	
	net.eval()
	test_loss = 0
	correct_pred = 0
	
	with torch.no_grad():
		
		for x, label in test_loader:
			
			x, label = x.to(device), label.to(device)
			y = net(x)

			if loss_f == "mse":
				label = F.one_hot(label, 10).float()
				test_loss += F.mse_loss(y, label)
				correct_pred += (y.argmax(dim=1) == label.argmax(dim=1)).sum().item()
			if loss_f == "ce":
				test_loss += F.cross_entropy(y, label).item()
				pred = y.argmax(dim=1)
				correct_pred += (pred == label).sum().item()

			if mode=="snn":
				functional.reset_net(net)

	test_acc = 100. * correct_pred / len(test_loader.dataset)
	test_loss /= len(test_loader) 
	
	print("===> Test Accuracy : {:.2f}%, Test Average loss: {:.8f}".format(test_acc, test_loss))
	return test_loss, test_acc


def save_model(net, path):
	torch.save(net.state_dict(), path)

