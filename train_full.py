import torch, json, os, pickle, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from spikingjelly.clock_driven import neuron, functional, surrogate, layer

import models

config_file_path = "config.json"
logs_path = os.path.join("logs","train_full")
save_path = os.path.join("saved_models","train_full")

if not os.path.exists(logs_path):
    os.mkdir("logs")
    os.mkdir(logs_path)

if not os.path.exists(save_path):
    os.mkdir("saved_models")
    os.mkdir(save_path)

with open(config_file_path) as f:
    config = json.load(f)

data_root = config["data_root"]
seed = config["seed"]
batch_size = config["batch_size"]

log_interval = 60000//(4*batch_size) # in order to have 5 logs in each epoch depending on the batch_size
mnist_mean = 0.1307
mnist_std = 0.3081

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train_full(net, mode, train_loader, optimizer, device, epoch):
    
    net.train()
    correct_pred = 0
    train_loss = 0

    for batch_idx, (x, label) in enumerate(train_loader):
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        y = net(x)
        label = F.one_hot(label, 10).float()
        loss = F.mse_loss(y, label)
        correct_pred += (y.argmax(dim=1) == label.argmax(dim=1)).sum().item()
        #loss = F.cross_entropy(y,label)
        #pred = y.argmax(dim=1)
        #correct_pred += (pred == label).sum().item()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        if mode=="snn":
            functional.reset_net(net)

    train_acc = 100. * correct_pred / len(train_loader.dataset)
    train_loss /= len(train_loader)

    print("\n===> Train Epoch Accuracy : {:.2f}%, , Train Average loss: {:.8f}".format(train_acc, train_loss))
    return train_loss, train_acc

def test_full(net, mode, test_loader, device):
    net.eval()
    test_loss = 0
    correct_pred = 0
    with torch.no_grad():
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            y = net(x)
            label = F.one_hot(label, 10).float()
            loss = F.mse_loss(y, label)
            correct_pred += (y.argmax(dim=1) == label.argmax(dim=1)).sum().item()
            #test_loss += F.cross_entropy(y, label).item()
            #pred = y.argmax(dim=1)
            #correct_pred += (pred == label).sum().item()

            if mode=="snn":
                functional.reset_net(net)

    test_acc = 100. * correct_pred / len(test_loader.dataset)
    test_loss /= len(test_loader) 
    print("===> Test Accuracy : {:.2f}%, Test Average loss: {:.8f}".format(test_acc, test_loss))

    return test_loss, test_acc

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

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size, shuffle=True, worker_init_fn=np.random.seed(0),num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size, worker_init_fn=np.random.seed(0),num_workers=0)


    if config["train_ann"]:
        config_ann = config["ann"]

        net = models.ANN().to(device)
        optimizer = optim.Adam(net.parameters(), lr = config_ann["lr"])

        epochs = config_ann["epochs"]
        best_acc = 0
        print("########## Training ANN for {} Epochs ##########\n".format(epochs))
        ann_logs = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}
        for epoch in range(epochs):
            
            train_loss, train_acc = train_full(net, "ann", train_loader, optimizer, device, epoch+1)
            test_loss, test_acc = test_full(net, "ann", test_loader, device)
            
            print("------------------------------------------------------")
            ann_logs["train_loss"].append(train_loss)
            ann_logs["train_acc"].append(train_acc)
            ann_logs["test_loss"].append(test_loss)
            ann_logs["test_acc"].append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(net.state_dict(), os.path.join(save_path, "ann.pth")) 

        with open(os.path.join(logs_path,"ann_logs.pickle"), "wb") as file:
            pickle.dump(ann_logs, file)
    

    if config["train_snn"]:
        config_snn = config["snn"]

        net = models.SNN().to(device)
        optimizer = optim.Adam(net.parameters(), lr = config_snn["lr"])

        epochs = config_snn["epochs"]
        best_acc = 0
        print("########## Training SNN for {} Epochs ##########\n".format(epochs))
        snn_logs = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}
        for epoch in range(epochs):
            
            train_loss, train_acc = train_full(net, "snn", train_loader, optimizer, device, epoch+1)
            test_loss, test_acc = test_full(net, "snn", test_loader, device)
            
            print("------------------------------------------------------")
            snn_logs["train_loss"].append(train_loss)
            snn_logs["train_acc"].append(train_acc)
            snn_logs["test_loss"].append(test_loss)
            snn_logs["test_acc"].append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(net.state_dict(), os.path.join(save_path, "snn.pth")) 

        with open(os.path.join(logs_path,"snn_logs.pickle"), "wb") as file:
            pickle.dump(snn_logs, file)