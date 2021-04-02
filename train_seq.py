import torch, json, os, pickle, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from spikingjelly.clock_driven import neuron, functional, surrogate, layer

import models, dataset, utils, func


config_file_path = "config.json"
with open(config_file_path) as f:
	config = json.load(f)

config = config["train_seq"]
data_root = config["data_root"]
seed = config["seed"]
batch_size = config["batch_size"]
save = config["save_best_weights"]
labels_phase_1 = config["labels_phase_1"]
labels_phase_2 = config["labels_phase_2"]
dropout = config["dropout"]
lr1 = config["lr_phase_1"]
lr2 = config["lr_phase_2"]
epochs1 = config["epochs_phase_1"]
epochs2 = config["epochs_phase_2"]
freeze_conv1 = config["freeze_conv1"]
freeze_conv2 = config["freeze_conv2"]
loss_ann = config["loss_ann"]
loss_snn = config["loss_snn"]
T = config["snn_T"]

mnist_mean = 0.1307
mnist_std = 0.3081

logs_path, save_path = utils.check_dirs(logs_path="train_seq", save_path="train_seq")
utils.set_seed(seed)


if __name__ == '__main__':

	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		print("Training on {} !".format(torch.cuda.get_device_name()))
	else:
		print("Training on CPU :( ")


	dataset_train_1 = dataset.dataset_prepare(labels_phase_1, data_root, train=True)
	dataset_test_1 = dataset.dataset_prepare(labels_phase_1, data_root, train=False)
	dataset_train_2 = dataset.dataset_prepare(labels_phase_2, data_root, train=True)
	dataset_test_2 = dataset.dataset_prepare(labels_phase_2, data_root, train=False)

	train_loader_1 = torch.utils.data.DataLoader(dataset_train_1, batch_size, shuffle=True)#, worker_init_fn=np.random.seed(0),num_workers=0)
	test_loader_1 = torch.utils.data.DataLoader(dataset_test_1, batch_size)#, worker_init_fn=np.random.seed(0),num_workers=0)
	train_loader_2 = torch.utils.data.DataLoader(dataset_train_2, batch_size, shuffle=True)#, worker_init_fn=np.random.seed(0),num_workers=0)
	test_loader_2 = torch.utils.data.DataLoader(dataset_test_2, batch_size)#, worker_init_fn=np.random.seed(0),num_workers=0)


	if config["train_ann"]:
		
		net = models.ANN(dropout=dropout).to(device)
		optimizer = optim.Adam(net.parameters(), lr = lr1)

		print("\n################ Training Phase 1 - ANN for {} Epochs ################\n".format(epochs1))
		ann_logs = {"train_acc_1":[], "train_acc_2":[],
					"test_acc_1":[], "test_acc_2":[]}
		best_acc = 0
		for epoch in range(epochs1):
			print("============ ANN1 - epoch {} / {}".format(epoch, epochs1))
			_, train_acc = func.train(net, "ann", train_loader_1, optimizer, device, epoch+1,loss_f=loss_ann)
			_, test_acc = func.test(net, "ann", test_loader_1, device, loss_f=loss_ann)
			
			print("------------------------------------------------------")
			ann_logs["train_acc_1"].append(train_acc)
			ann_logs["test_acc_1"].append(test_acc)


		if freeze_conv1:
			for param in net.convLayer1.parameters():
				param.requires_grad = False
		if freeze_conv2:
			for param in net.convLayer2.parameters():
	  			param.requires_grad = False

		optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = lr2)
		print("\n################ Training Phase 2 - ANN for {} Epochs ################\n".format(epochs2))
		best_acc = 0
		for epoch in range(epochs2):
			print("============ ANN2 - epoch {} / {}".format(epoch, epochs2))
			_, train_acc = func.train(net, "ann", train_loader_2, optimizer, device, epoch+1, loss_f=loss_ann)
			_, test_acc_2 = func.test(net, "ann", test_loader_2, device, loss_f=loss_ann)
			_, test_acc_1 = func.test(net, "ann", test_loader_1, device, loss_f=loss_ann) 
			
			print("---------------------------------------------------------")
			ann_logs["train_acc_2"].append(train_acc)
			ann_logs["test_acc_2"].append(test_acc_2)
			ann_logs["test_acc_1"].append(test_acc_1)

			if save and test_acc_2 + test_acc_1> best_acc:
				best_acc = test_acc_2 + test_acc_1
				torch.save(net.state_dict(), os.path.join(save_path, "ann.pth"))

		with open(os.path.join(logs_path,"ann_logs.pickle"), "wb") as file:
			pickle.dump(ann_logs, file)
	

	if config["train_snn"]:

		net = models.SNN(T=T, dropout=dropout).to(device)
		optimizer = optim.Adam(net.parameters(), lr = lr1)

		print("########## Training Phase 1 - SNN for {} Epochs ##########\n".format(epochs1))
		snn_logs = {"train_acc_1":[], "train_acc_2":[],
					"test_acc_1":[], "test_acc_2":[]}
		best_acc = 0
		for epoch in range(epochs1):
			print("============ SNN1 - epoch {} / {}".format(epoch, epochs1))
			_, train_acc = func.train(net, "snn", train_loader_1, optimizer, device, epoch+1,loss_f=loss_snn)
			_, test_acc = func.test(net, "snn", test_loader_1, device, loss_f=loss_snn)
			
			print("------------------------------------------------------")
			snn_logs["train_acc_1"].append(train_acc)
			snn_logs["test_acc_1"].append(test_acc)


		if freeze_conv1:
			for param in net.static_conv.parameters():
				param.requires_grad = False
		if freeze_conv2:
			for param in net.conv.parameters():
	  			param.requires_grad = False

		optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = lr2)
		print("\n########## Training Phase 2 - SNN for {} Epochs ############\n".format(epochs2))
		best_acc = 0
		for epoch in range(epochs2):
			print("============ SNN2 - epoch {} / {}".format(epoch, epochs2))
			_, train_acc = func.train(net, "snn", train_loader_2, optimizer, device, epoch+1, loss_f=loss_snn)
			_, test_acc_2 = func.test(net, "snn", test_loader_2, device, loss_f=loss_snn)
			_, test_acc_1 = func.test(net, "snn", test_loader_1, device, loss_f=loss_snn) 
			
			print("---------------------------------------------------------")
			snn_logs["train_acc_2"].append(train_acc)
			snn_logs["test_acc_2"].append(test_acc_2)
			snn_logs["test_acc_1"].append(test_acc_1)

			if save and test_acc_2 + test_acc_1> best_acc:
				best_acc = test_acc_2 + test_acc_1
				torch.save(net.state_dict(), os.path.join(save_path, "snn.pth"))

		with open(os.path.join(logs_path,"snn_logs.pickle"), "wb") as file:
			pickle.dump(snn_logs, file)