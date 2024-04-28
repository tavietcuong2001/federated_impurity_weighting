# Installing dependencies

import torch
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler

import flwr as fl
from flwr.common import ndarrays_to_parameters

import utils
from fedalg import FedAvg, FedProx, FedAdp, FedImp
from model import MLP, CNN, CNN2, CNN4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")


# Loading the data

NUM_CLIENTS = 10
NUM_IIDS = 3
BATCH_SIZE = 100

trainset, testset = utils.load_data("cifar10")
ids, dist = utils.partition_data(trainset, num_clients=NUM_CLIENTS, num_iids=NUM_IIDS, alpha=100, beta=0.01)
#ids, dist = utils.partition_data_special_case(trainset, num_clients=NUM_CLIENTS, num_iids=NUM_IIDS)

for i in range(NUM_CLIENTS):
    print(f"Client {i+1}: {dist[i]}")

entropies = [utils.compute_entropy(dist[i]) for i in range(NUM_CLIENTS)]

trainloaders = []
valloaders = []
val_length = [int(len(testset)/NUM_CLIENTS)] * NUM_CLIENTS
valsets = random_split(testset, val_length)
for i in range(NUM_CLIENTS):
    trainloaders.append(DataLoader(trainset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(ids[i])))
    valloaders.append(DataLoader(valsets[i], batch_size=BATCH_SIZE))


# Defining Flower client

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return utils.get_parameters(self.net)

    def fit(self, parameters, config):
        utils.set_parameters(self.net, parameters)
        loss, accuracy = utils.train(self.net, self.trainloader, learning_rate=config["learning_rate"])
        #loss, accuracy = utils.train(self.net, self.trainloader, learning_rate=config["learning_rate"], proximal_mu=config["proximal_mu"])
        return utils.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid}

    def evaluate(self, parameters, config):
        utils.set_parameters(self.net, parameters)
        loss, accuracy = utils.test(self.net, self.valloader)
        return loss, len(self.valloader.sampler), {"accuracy": accuracy}


def client_fn(cid: str) -> FlowerClient:
    net = CNN2().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader).to_client()


# Training

NUM_ROUNDS = 800
current_parameters = ndarrays_to_parameters(utils.get_parameters(CNN2()))
client_resources = {"num_cpus": 2, "num_gpus": 0.1} if DEVICE.type == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}

fl.simulation.start_simulation(
    client_fn = client_fn,
    num_clients = NUM_CLIENTS,
    config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy = FedAdp(num_rounds=NUM_ROUNDS, num_clients=NUM_CLIENTS, current_parameters=current_parameters),
    client_resources = client_resources
)