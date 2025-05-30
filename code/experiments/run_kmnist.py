import sys
import os
from enum import Enum
from tqdm import tqdm
from time import time

sys.path.append(os.path.abspath("code"))

import layers, models
from geoopt.optim import RiemannianSGD, RiemannianAdam

import numpy as np

import torch
from torch import nn
import pandas as pd

from itertools import product

import torchvision
from torchvision.datasets import MNIST, KMNIST, FashionMNIST, CIFAR10
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from itertools import *
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dataset(Enum):
    mnist = 'mnist'
    kmnist = 'kmnist'
    fmnist = 'fmnist'
    cifar10 = 'cifar10'


datasetdict = {
    dataset.mnist: MNIST,
    dataset.kmnist: KMNIST,
    dataset.fmnist: FashionMNIST,
    dataset.cifar10: CIFAR10,
}

transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Lambda(lambda x: torch.flatten(x))
                             ])

p = Path('campaign_path')

class model_type(Enum):
    euclidean = 'eucl'
    hyperbolic = 'hyp'
    eubn = 'euclbn'
    poincare = 'poincare'
    lorentz = 'lorentz'


model_dict = {
    model_type.hyperbolic : models.HyperbolicNetwork,
    model_type.euclidean : models.EuclideanNetwork,
    model_type.poincare: models.PoincareNetwork,
}

head_dict = {
   model_type.hyperbolic : layers.HyperbolicRegressionLayer,
   model_type.euclidean : torch.nn.Linear,
   model_type.poincare: torch.nn.Linear,
}
    
class activation(Enum):
    identity = 'id'
    relu = 'relu'
    leaky_relu = 'lerelu'
    sigmoid = 'sigmoid'
    dmelu = 'dmelu'

activation_dict = {
    activation.identity: nn.Identity,
    activation.relu: nn.ReLU,
    activation.leaky_relu: nn.LeakyReLU,
    activation.sigmoid: nn.Sigmoid,
    activation.dmelu: layers.DmELU
}

configs = [
    {'mtype': x,
     'activation': y,
     'learning_rate':z,
     'dataset': w,
     'nneurons': nn,
     'nlayers': nl} for (x,y),z, w, nn, nl in
     product(
        [
        (model_type.euclidean, activation.dmelu),
        (model_type.hyperbolic, activation.dmelu),
        (model_type.hyperbolic, activation.identity),
        (model_type.poincare, activation.dmelu)
        ],
        [1e-4],
        [
           dataset.kmnist,
        ],
        [20, 40,100, 500], #nneurons
        [1,2,3,4,5,6,7]
    )
]

base_path = Path('data/kmnist')
base_path.mkdir(exist_ok = True, parents=True)
reps = 10
epochs = 1000
measure_cartan = False
save_model = False

def path_from_config(config):
    return base_path / config['dataset'].value / (config['mtype'].value + '_' + config['activation'].value + '_' + str(config['nneurons']) + '_' + str(config['nlayers']) + '_' + str(config['learning_rate'])[2:6])


def train(config, path, seed):
    start = time()
    es = 15
    loss_buffer = np.inf * np.ones(es)
    lr = config['learning_rate']
    wd = 1e-5
    nlayers = config['nlayers']
    torch.manual_seed(seed)
    criterion = nn.CrossEntropyLoss()
    rows = []

    train_dataset = datasetdict[config['dataset']](root='files', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=True,
                              drop_last=True,
                              num_workers=32,
                              pin_memory=True)
    test_dataset = datasetdict[config['dataset']](root='files', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset,
                              batch_size=len(test_dataset),
                              shuffle=True,
                              drop_last=True)
    
    input_dim = iter(train_loader).__next__()[0].size()[1]
    output_dim = len(train_dataset.classes)
    nneurons = config['nneurons']
    model = model_dict[config['mtype']](
       input_dim,
       activation_dict[config['activation']],
       [nneurons]*nlayers,
       head=head_dict[config['mtype']](nneurons, output_dim)
       )
    model.to(device)
    pbar = tqdm(range(epochs))
    optimizer = RiemannianAdam(params = model.parameters(), lr=lr, weight_decay=wd)
    
    for epoch in pbar:
      train_hyper = None
      model.train()
      train_losses = []
      train_accuracies = []
      for x_train, y_train in train_loader:
        x_train = x_train.flatten(start_dim=1).to(device)
        y_train = y_train.to(device)
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()

        train_accuracies.append(torch.mean((torch.argmax(outputs.detach(), dim=-1)== y_train.squeeze()).float()).detach().cpu())
        train_losses.append(loss.detach().cpu())
        if measure_cartan:
            if train_hyper is None:
                train_hyper = [[x.item()] for x in model.distance_from_euclidean(x_train)]
            else:
                train_hyper = [past + [x.item()] for past, x in zip(train_hyper, model.distance_from_euclidean(x_train))]
        
        optimizer.step()
        
      with torch.no_grad():
        for x_test, y_test in test_loader:
          x_test = x_test.flatten(start_dim=1).to(device)
          y_test = y_test.to(device)

          test_accuracy = torch.mean((torch.argmax(model(x_test), dim=-1).detach()== y_test.squeeze()).float()).detach().cpu()
          test_loss = criterion(model(x_test), y_test)
          train_loss = np.mean(train_losses)
          train_accuracy = np.mean(train_accuracies)
          if measure_cartan:
            test_hyperbolicity = [x.item() for x in model.distance_from_euclidean(x_test)]
            train_hyperbolicity = [np.mean(x) for x in train_hyper]
          else:
             test_hyperbolicity = None
             train_hyperbolicity = None
          rows.append([
             config['mtype'],
             config['activation'],
             test_accuracy.cpu().numpy(),
             test_loss.cpu().numpy(),
             train_accuracy,
             train_loss,
             epoch,
             lr,
             wd,
             config['nneurons'],
             nlayers,
             train_hyperbolicity,
             test_hyperbolicity,
             time()-start])
          pbar.set_description(f"Train loss: {train_loss} ! Test accuracy: {test_accuracy.cpu().numpy()}")
          break

      loss_buffer[:-1] = loss_buffer[1:]
      tl = test_loss.detach().cpu().numpy()
      loss_buffer[-1] = tl
      if np.all(loss_buffer[-1]<loss_buffer[:-1]):
        best_model_index = es-1
        if save_model:
            torch.save(model, path / (str(seed) + '_checkpoint.pt'))
      else:
        best_model_index -=1
      if not best_model_index:
        break
      
    
    df = pd.DataFrame(rows, columns = [
       'Model',
       'Activation',
       'Test accuracy',
       'Test loss',
       'Train accuracy',
       'Train loss',
       'Epoch',
       'Learning rate',
       'Weight decay',
       'Neurons',
       'Nlayers',
       'Train hyperbolicities',
       'Test hyperbolicities',
       'Time'
       ])
    df.to_csv(path / (str(seed) + '_data.csv'))

if __name__ == '__main__':
    for config in configs:
        path = path_from_config(config)
        try:
            path.mkdir(exist_ok=False, parents=True)
            for i in range(reps):
                seed = np.random.randint(0, 10000)
                train(config, path, seed)
        except FileExistsError:
            pass
