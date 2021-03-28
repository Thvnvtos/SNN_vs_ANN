import torch
from torchvision import datasets, transforms

mnist_mean = 0.1307
mnist_std = 0.3081

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