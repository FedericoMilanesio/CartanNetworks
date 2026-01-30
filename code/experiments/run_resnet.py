import sys
import os
from enum import Enum
from tqdm import tqdm
from time import time
import pdb

import itertools

sys.path.append(os.path.abspath("code"))

import layers, models, resnet
from geoopt.optim import RiemannianSGD
import geoopt
from geoopt.manifolds import Sphere

sys.path.append(os.path.abspath("HyperbolicCV/code"))
from classification.models.classifier import ResNetClassifier

from torchvision.models.resnet import ResNet, BasicBlock

class ResNet18(ResNet):
    def __init__(self, block=BasicBlock, layers=[2,2,2,2], num_classes = 1000, zero_init_residual = False, groups = 1, width_per_group = 64, replace_stride_with_dilation = None, norm_layer = None, input_shape=None):
       super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)

class FullyResNet(ResNetClassifier):
       def __init__(self, 
            num_classes = 200
        ):
        super().__init__(
            num_layers=18,
            enc_type = 'lorentz', 
            dec_type = 'lorentz',
            enc_kwargs={'k':1.},
            dec_kwargs={
               'clip_r' : 4.,
                'embed_dim' : 512,
                'num_classes' : num_classes,
                'type':'mlr',
                'k':1.,
                'learn_k':True
            }
            )

import numpy as np

import torch
from torch import nn
import pandas as pd

from itertools import product

import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, CelebA, ImageFolder
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from itertools import *
import numpy as np

class CelebAThreeAttrClassification(CelebA):
    def __init__(self, train, *args, **kwargs):
        if train:
           split = 'train'
        else:
            split = 'test'
        super().__init__(*args, **kwargs, split=split)

        self.selected_attrs = ['Male', 'Young', 'Smiling']
        self.attr_indices = [self.attr_names.index(attr) for attr in self.selected_attrs]

    def __getitem__(self, idx):
        image, attributes = super().__getitem__(idx)

        binary_attrs = [(attributes[i].item() == 1) for i in self.attr_indices]
        binary_attrs = [int(val) for val in binary_attrs]

        class_label = binary_attrs[0] * 4 + binary_attrs[1] * 2 + binary_attrs[2]

        return image, torch.tensor(class_label, dtype=torch.long)

class TinyImagenet224(ImageFolder):
   def __init__(self, root='files/', train=True, transform = None, target_transform = None, is_valid_file = None, allow_empty = False):
      super().__init__(root + ('tiny-224/train' if train else 'tiny-224/test'), transform, target_transform, is_valid_file = is_valid_file, allow_empty = allow_empty)

class TinyImagenet(ImageFolder):
   def __init__(self, root='files/', train=True, transform = None, target_transform = None, is_valid_file = None, allow_empty = False):
      super().__init__(root + ('tiny-imagenet-200/train' if train else 'tiny-imagenet-200/test'), transform, target_transform, is_valid_file = is_valid_file, allow_empty = allow_empty)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dataset(Enum):
    cifar10 = 'cifar10'
    cifar100 = 'cifar100'
    celebA = 'celebA'
    tinyimagenet = 'tiny_imagenet'
    tinyimagenet224 = 'tiny_imagenet_224'




datasetdict = {
    dataset.cifar10: CIFAR10,
    dataset.cifar100: CIFAR100,
    dataset.celebA: CelebAThreeAttrClassification,
    dataset.tinyimagenet: TinyImagenet,
    dataset.tinyimagenet224: TinyImagenet224
}

dataset_channels = {
    dataset.cifar10: 3,
    dataset.cifar100: 3,
    dataset.celebA: 3,
    dataset.tinyimagenet: 3
}

dataset_input_sizes = {
    dataset.cifar10: (32,32),
    dataset.cifar100: (32,32),
    dataset.celebA: (224,224),
    dataset.tinyimagenet: (64, 64),
    dataset.tinyimagenet224: (224, 224)
}

dataset_classes = {
    dataset.cifar10: 10,
    dataset.cifar100: 100,
    dataset.celebA: 8,
    dataset.tinyimagenet: 200,
    dataset.tinyimagenet224: 200
}

dataset_norms = {
   dataset.cifar10:{'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
   dataset.cifar100:{'mean': [0.5071, 0.4865, 0.4409], 'std': [0.2673, 0.2564, 0.2762]},
   dataset.celebA:{'mean': [0.506, 0.425, 0.384], 'std': [0.266, 0.245, 0.241]},
   dataset.tinyimagenet224:{'mean': [0.48051679134368896, 0.44834452867507935, 0.3978164494037628], 'std':[0.2666704058647156, 0.2584282457828522, 0.27223914861679077]},
   dataset.tinyimagenet:{'mean': [0.48024600744247437, 0.44807225465774536, 0.39754778146743774], 'std':[0.2769860327243805, 0.26906439661979675, 0.28208184242248535]}
}

p = Path('campaign_path')

class model_type(Enum):
    halexnet = 'halexnet'
    alexnet = 'alexnet'
    hresnet18 = 'hresnet18'
    hresnet34 = 'hresnet34'
    resnet18 = 'resnet18'
    resnet34 = 'resnet34'
    fresnet18 = 'fresnet18'

class HResNet34(resnet.HResNet):
    def __init__(self,
                 block=resnet.HBasicBlock,
                 layers=[3,4,6,3],
                 num_classes = 1000,
                 zero_init_residual = False,
                 groups = 1,
                 width_per_group = 64, 
                 replace_stride_with_dilation = None,
                 norm_layer = None,
                 input_shape = (3, 64, 64)):
       super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer, input_shape)

class HResNet18(resnet.HResNet):
    def __init__(self,
                 block=resnet.HBasicBlock,
                 layers=[2,2,2,2],
                 num_classes = 1000,
                 zero_init_residual = False,
                 groups = 1,
                 width_per_group = 64, 
                 replace_stride_with_dilation = None,
                 norm_layer = None,
                 input_shape = (3, 64, 64)
                 ):
       super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer, input_shape)


model_dict = {
    model_type.halexnet: models.HyperAlexNetCifar,
    model_type.alexnet: models.AlexNetCifar,
    model_type.hresnet18: HResNet18,
    model_type.hresnet34: HResNet34,
    model_type.fresnet18: FullyResNet,
    model_type.resnet18: ResNet18
}

configs = [
    {'mtype': x,
     'learning_rate':z,
     'dataset': w} for x,z,w in
     product(
        [
           model_type.hresnet18,
           model_type.resnet18
        ],
        np.logspace(-1, -3, 5)[:3],
       [
         dataset.tinyimagenet224
        ])
]

base_path = Path('data/all_conv_aug')
base_path.mkdir(exist_ok = True, parents=True)
reps = 1
epochs = 1
save_model = False

def path_from_config(config):
    return base_path / (config['dataset'].value) / (config['mtype'].value + str(config['learning_rate'])[2:6])


def train(config, path, seed):
    start = time()
    es = 15
    loss_buffer = np.inf * np.ones(es)
    lr = config['learning_rate']
    wd = 0.0005
    torch.manual_seed(seed)
    criterion = nn.CrossEntropyLoss()
    rows = []
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**dataset_norms[config['dataset']])
    ])
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**dataset_norms[config['dataset']])
    ])

    train_dataset = datasetdict[config['dataset']](root='files/', train=True, transform=transform_train,)
    train_loader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=True,
                              drop_last=True,
                              num_workers=2,
                              pin_memory=True)
    test_dataset = datasetdict[config['dataset']](root='files/', train=False, transform=transform_test,)
    test_loader = DataLoader(test_dataset,
                              batch_size = 64,
                              shuffle=True,
                              drop_last=True)
    
    
    model = model_dict[config['mtype']](num_classes = dataset_classes[config['dataset']], input_shape = (3, *dataset_input_sizes[config['dataset']]))
    model.to(device)
    pbar = tqdm(range(epochs))

    fc_params = []
    body_params = []
    
    for name, module in model.named_modules():
        if isinstance(module, layers.HyperbolicRegression):
            fc_params.extend(module.parameters())
    
    fc_param_ids = set(id(p) for p in fc_params)
    for p in model.parameters():
        if id(p) not in fc_param_ids:
            body_params.append(p)
    
    param_groups = [
        {'params': body_params, 'lr': lr},
        {'params': fc_params, 'lr': min(lr, 1e-3)}
    ]
    
    optimizer = RiemannianSGD(param_groups,lr=lr, weight_decay=wd)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, cooldown=10)
    
    for epoch in pbar:
      model.train()
      train_losses = []
      train_accuracies = []
      for x_train, y_train in train_loader:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        if torch.isnan(loss):
           raise Exception("Loss is nan")


        train_accuracies.append(torch.mean((torch.argmax(outputs.detach(), dim=-1)== y_train.squeeze()).float()).detach().cpu())
        train_losses.append(loss.detach().cpu())
        optimizer.step()
        
      with torch.no_grad():
        model.eval()
        train_loss = np.mean(train_losses)
        train_accuracy = np.mean(train_accuracies)

        test_losses = []
        test_accuracies = []

        for x_test, y_test in test_loader:
          x_test = x_test.to(device)
          y_test = y_test.to(device)

          test_accuracies.append(torch.mean((torch.argmax(model(x_test), dim=-1).detach()== y_test.squeeze()).float()).detach().cpu().item())
          test_losses.append(criterion(model(x_test), y_test).cpu().item())
        test_accuracy = sum(test_accuracies)/len(test_accuracies)
        test_loss = sum(test_losses)/len(test_losses)
        rows.append([
            config['mtype'],
            test_accuracy,
            test_loss,
            train_accuracy,
            train_loss,
            epoch,
            lr,
            wd,
            time()-start])
        pbar.set_description(f"Train loss: {train_loss:.3f} ! Test accuracy: {test_accuracy:.3f} | LR : {scheduler.get_last_lr()[-1]:.3f}")
        scheduler.step(test_loss)

      loss_buffer[:-1] = loss_buffer[1:]

      loss_buffer[-1] = test_loss
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
       'Test accuracy',
       'Test loss',
       'Train accuracy',
       'Train loss',
       'Epoch',
       'Learning rate',
       'Weight decay',
       'Time'
       ])
    df.to_csv(path / (str(seed) +  '_data.csv'))

if __name__ == '__main__':
    #task_id = int(os.environ['TASK_ID'])
    task_id = 0
    print(task_id)
    l = list(configs)
    print(len(l[task_id::20]))
    for config in l:
        path = path_from_config(config)
        path.mkdir(exist_ok=True, parents=True)
        done = len(list(path.glob('*.csv')))
        print(config)
        try:
            if done<reps:
                for i in range(reps - done):
                    seed = np.random.randint(0, 10000)
                    print(seed)
                    train(config, path, seed)
        except Exception as e:
            print(e)
            print("config interrupted")
            print(config)
