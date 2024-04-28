import numpy as np
from collections import Counter, OrderedDict
from typing import List, Dict
import random
import math
import copy

import torch
from torch.distributions.dirichlet import Dirichlet
from torchvision.datasets import CIFAR10, EMNIST
import torchvision.transforms as transforms
import torch.nn as nn
from torch.optim import SGD

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def renormalize(dist: torch.tensor, labels: List[int], label: int):
    idx = labels.index(label)
    dist[idx] = 0
    dist /= sum(dist)
    dist = torch.concat((dist[:idx], dist[idx+1:]))
    return dist


def load_data(dataset: str):
    if dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = CIFAR10("data", train=True, download=True, transform=train_transform)
        testset = CIFAR10("data", train=False, download=True, transform=test_transform)
    
    elif dataset == "emnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        trainset = EMNIST("data", split="balanced", train=True, download=True, transform=transform)
        testset = EMNIST("data", split="balanced", train=False, download=True, transform=transform)
    
    return trainset, testset


def partition_data(trainset, num_clients: int, num_iids: int, alpha: float, beta: float):
    classes = trainset.classes
    client_size = int(len(trainset)/num_clients)
    label_size = int(len(trainset)/len(classes))
    data = list(map(lambda x: (trainset[x][1], x), range(len(trainset))))
    data.sort()
    data = list(map(lambda x: data[x][1], range(len(data))))
    data = [data[i*label_size:(i+1)*label_size] for i in range(len(classes))]

    ids = [[] for _ in range(num_clients)]
    label_dist = []
    labels = list(range(len(classes)))

    for i in range(num_clients):
        concentration = torch.ones(len(labels))*alpha if i < num_iids else torch.ones(len(labels))*beta
        dist = Dirichlet(concentration).sample()
        for _ in range(client_size):
            label = random.choices(labels, dist)[0]
            id = random.choices(data[label])[0]
            ids[i].append(id)
            data[label].remove(id)

            if len(data[label]) == 0:
                dist = renormalize(dist, labels, label)
                labels.remove(label)

        counter = Counter(list(map(lambda x: trainset[x][1], ids[i])))
        label_dist.append({classes[i]: counter.get(i) for i in range(len(classes))})

    return ids, label_dist


def partition_data_special_case(trainset, num_clients: int, num_iids: int):
    classes = trainset.classes
    client_size = int(len(trainset)/num_clients)
    label_size = int(len(trainset)/len(classes))
    data = list(map(lambda x: (trainset[x][1], x), range(len(trainset))))
    data.sort()
    data = list(map(lambda x: data[x][1], range(len(data))))
    
    grouped_data = [data[i*label_size:(i+1)*label_size] for i in range(len(classes))]
    non_iid_labels = random.sample(range(len(classes)), 2) if len(classes) == 10 else list(range(10))
    non_iid_data = []
    for label in non_iid_labels:
        non_iid_data += grouped_data[label]

    ids = []
    label_dist = []
    for i in range(num_clients):
        temp_data = data if i < num_iids else non_iid_data
        id = random.sample(temp_data, client_size)
        ids.append(id)
        
        counter = Counter(list(map(lambda x: trainset[x][1], ids[i])))
        label_dist.append({classes[i]: counter.get(i) for i in range(len(classes))})

    return ids, label_dist


def compute_entropy(counts: Dict):
    entropy = 0.0
    counts = list(counts.values())
    counts = [0 if value is None else value for value in counts]
    for value in counts:
        entropy += -value/sum(counts) * math.log(value/sum(counts), len(counts)) if value != 0 else 0
    return entropy


def train(net, trainloader, learning_rate: float, proximal_mu: float = None):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=learning_rate)
    net.train()
    running_loss, running_corrects = 0.0, 0
    global_params = copy.deepcopy(net).parameters()
    
    for images, labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(images)

        if proximal_mu != None:
            proximal_term = 0.0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)
            loss = criterion(outputs, labels) + (proximal_mu / 2) * proximal_term
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * images.shape[0]
        running_corrects += torch.sum(predicted == labels).item()

    running_loss /= len(trainloader.sampler)
    acccuracy = running_corrects / len(trainloader.sampler)
    return running_loss, acccuracy


def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    corrects, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            predicted = torch.argmax(outputs, dim=1)
            loss += criterion(outputs, labels).item() * images.shape[0]
            corrects += torch.sum(predicted == labels).item()
    loss /= len(testloader.sampler)
    accuracy = corrects / len(testloader.sampler)
    return loss, accuracy


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict)