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

class SimpleRegression(Dataset):
    def __init__(self, root=None, train=True, transform=None, download=False, dimensions=10):
       super().__init__()
       self.size = int(200) if train else int(1000)
       self.data = 2*torch.rand((self.size, dimensions))-1
       self.label = self.tolearn(self.data) + torch.randn(self.size) * 0.01 * train

    def __len__(self):
       return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def tolearn(self, data):
        raise NotImplementedError
    
class SINC(SimpleRegression):
    def tolearn(self, data):
       n = torch.linalg.norm(data, dim=-1)
       return torch.sin(n)/n

class SINC3(SimpleRegression):
    def tolearn(self, data):
       n = torch.linalg.norm(data, dim=-1, ord=3)
       return torch.sin(n)/n

class Prod2(SimpleRegression):
    def tolearn(self, data):
        return data[..., 0] + data[..., 0] * data[..., 1]
    
class Prod3(SimpleRegression):
    def tolearn(self, data):
        return data[..., 0] + data[..., 0] * data[..., 1] + data[..., 0] * data[..., 1] * data[..., 2]


class Hyp(SimpleRegression):
    def tolearn(self, data):
        return (torch.sum(data[...,:-1]**2, axis=-1) - data[...,-1]**2)/data.shape[1]
class Norm(SimpleRegression):
    def tolearn(self, data):
        return torch.linalg.norm(data, dim=-1)
    
class Sine(SimpleRegression):
    def tolearn(self, data):
       n = torch.linalg.norm(data, dim=-1)
       return torch.sin(n)

class dataset(Enum):
    sinc = 'sinc'
    sinc3 = 'sinc3'
    sine = 'sine'
    norm = 'norm'
    prod2 = 'prod2'
    prod3 = 'prod3'
    hyp = 'hyp'


datasetdict = {
    dataset.sinc: SINC,
    dataset.sinc3: SINC3,
    dataset.sine: Sine,
    dataset.norm: Norm,
    dataset.prod2: Prod2,
    dataset.prod3: Prod3,
    dataset.hyp: Hyp
}

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
     'nlayers': nl,
     'problem_dimension': pd} for (x,y),z, w, nn, nl, pd in
     product(
        [
        (model_type.hyperbolic, activation.dmelu),
        (model_type.euclidean, activation.dmelu),
        (model_type.hyperbolic, activation.identity),
        (model_type.poincare, activation.dmelu),
        ],
        [1e-2],
        [
           dataset.sinc,
           dataset.sinc3,
           dataset.prod2,
           dataset.prod3,
           dataset.hyp,
        ],
        [20], #nneurons
        [2], #nlayers
        [10] #problem dimension if applicable
    )
]

base_path = Path('data/regression')
base_path.mkdir(exist_ok = True, parents=True)
reps = 10
epochs = 5000
measure_cartan = False
save_model = False

def path_from_config(config):
    return base_path / config['dataset'].value / (config['mtype'].value + '_' + config['activation'].value + '_' + str(config['nneurons']) + '_' + str(config['nlayers']) + '_' + str(config['problem_dimension']))


def train(config, path, seed):
    start = time()
    problem_dimension = config['problem_dimension']
    lr = config['learning_rate']
    wd = 1e-4
    nlayers = config['nlayers']
    torch.manual_seed(seed)
    criterion = nn.MSELoss()
    rows = []

    train_dataset = datasetdict[config['dataset']](root='files', train=True, download=True, dimensions=problem_dimension)
    train_loader = DataLoader(train_dataset,
                              batch_size=200,
                              shuffle=True,
                              drop_last=True)
    test_dataset = datasetdict[config['dataset']](root='files', train=False, download=True, dimensions=problem_dimension)
    test_loader = DataLoader(test_dataset,
                              batch_size=len(test_dataset),
                              shuffle=True,
                              drop_last=True)
    
    input_dim = iter(train_loader).__next__()[0].size()[1]
    output_dim = 1
    nneurons = config['nneurons']
    model = model_dict[config['mtype']](
       input_dim,
       activation_dict[config['activation']],
       [nneurons]*nlayers,
       head=head_dict[config['mtype']](nneurons, output_dim)
       )
    pbar = tqdm(range(epochs))
    optimizer = RiemannianAdam(params = model.parameters(), lr=lr, weight_decay=wd)
    
    for epoch in pbar:
      train_hyper = None
      model.train()
      train_losses = []
      for x_train, y_train in train_loader:
        x_train = x_train.flatten(start_dim=1)
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()

        train_losses.append(loss.detach().cpu())
        if measure_cartan:
            if train_hyper is None:
                train_hyper = [[x.item()] for x in model.distance_from_euclidean(x_train)]
            else:
                train_hyper = [past + [x.item()] for past, x in zip(train_hyper, model.distance_from_euclidean(x_train))]
        
        optimizer.step()
        
      with torch.no_grad():
        for x_test, y_test in test_loader:
          x_test = x_test.flatten(start_dim=1)
          y_test = y_test
          yvar = y_test.var()

          test_accuracy = None
          test_loss = criterion(model(x_test).squeeze(), y_test)
          train_loss = np.mean(train_losses)
          train_accuracy = None
          if measure_cartan:
            test_hyperbolicity = [x.item() for x in model.distance_from_euclidean(x_test)]
            train_hyperbolicity = [np.mean(x) for x in train_hyper]
          else:
             test_hyperbolicity = None
             train_hyperbolicity = None
          rows.append([
             config['mtype'],
             config['activation'],
             test_accuracy,
             test_loss.cpu().numpy(),
             test_loss.cpu().numpy()/yvar.cpu().numpy(),
             train_accuracy,
             train_loss,
             epoch,
             lr,
             wd,
             config['nneurons'],
             nlayers,
             train_hyperbolicity,
             test_hyperbolicity,
             problem_dimension,
             time() - start])
          pbar.set_description(f"Train loss: {train_loss:.4e}|Test loss: {test_loss.cpu().numpy():.4e}|NTL: {test_loss.cpu().numpy()/yvar:.4e}")
          break

    
    df = pd.DataFrame(rows, columns = [
       'Model',
       'Activation',
       'Test accuracy',
       'Test loss',
       'Normalized test loss',
       'Train accuracy',
       'Train loss',
       'Epoch',
       'Learning rate',
       'Weight decay',
       'Neurons',
       'Nlayers',
       'Train hyperbolicities',
       'Test hyperbolicities',
       'Problem dimension',
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
