import torch, random
import numpy as np
from torchvision import datasets, transforms

mnist_mean = 0.1307
mnist_std = 0.3081

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def dataset_prepare(targets,data_root, train):

	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((mnist_mean,), (mnist_std,))
		])

	dataset = datasets.MNIST(data_root, train=train, transform=transform)
	idx = [x in targets for x in dataset.targets]

	dataset.targets = dataset.targets[idx]
	dataset.data = dataset.data[idx]

	return dataset 


def dataset_prepare_fewshot(targets, k, data_root, train):

	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((mnist_mean,), (mnist_std,))
		])
	
	
	dataset_base = datasets.MNIST(data_root, train=train, transform=transform)
	dataset = datasets.MNIST(data_root, train=train, transform=transform)
	
	for t in targets:
		
		idx = [(x in [t]) for x in dataset_base.targets]
		valid_idxs = []
		for i in range(len(idx)):
			if idx[i]: valid_idxs.append(i)

		if len(dataset.targets) == len(dataset_base):
			k_idx = torch.randperm(len(valid_idxs))[:k]
			valid_idxs = valid_idxs[:k]
			dataset.targets = dataset_base.targets[valid_idxs]
			dataset.data = dataset_base.data[valid_idxs]
		else:
			k_idx = torch.randperm(len(valid_idxs))[:k]
			valid_idxs = valid_idxs[:k]
			dataset.targets = torch.cat((dataset.targets, dataset_base.targets[valid_idxs]),0)
			dataset.data = torch.cat((dataset.data, dataset_base.data[valid_idxs]),0)

	return dataset 