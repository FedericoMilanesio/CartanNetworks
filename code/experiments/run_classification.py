import os
import sys
import argparse
import yaml, json
from itertools import product
from enum import Enum
from itertools import product
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST, KMNIST, FashionMNIST, CIFAR10

sys.path.append(os.path.abspath("code"))
import models
import layers
from training import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = Path(os.getenv("OUTPUT_DIR", "data/classification"))
BASE_PATH.mkdir(exist_ok=True, parents=True)


class dataset(Enum):
    mnist = 'mnist'
    kmnist = 'kmnist'
    fmnist = 'fmnist'
    cifar10 = 'cifar10'

DATASET_DICT = {
    dataset.mnist: MNIST,
    dataset.kmnist: KMNIST,
    dataset.fmnist: FashionMNIST,
    dataset.cifar10: CIFAR10,
}

class ModelType(Enum):
    euclidean = 'eucl'
    hyperbolic = 'hyp'
    poincare = 'poincare'
    lorentz = 'lorentz'
    logr = 'logistic'
    h_plusplus = 'h_plusplus'

MODEL_DICT = {
    ModelType.hyperbolic: models.HyperbolicNetwork,
    ModelType.euclidean: models.EuclideanNetwork,
    ModelType.poincare: models.PoincareNetwork,
    ModelType.lorentz: models.LorentzNetwork,
    ModelType.h_plusplus: models.PlusPlusNetwork,
    ModelType.logr: lambda input_dim, act, layers, head: 10
}

class Activation(Enum):
    identity = 'id'
    relu = 'relu'
    leaky_relu = 'lerelu'
    sigmoid = 'sigmoid'
    dmelu = 'dmelu'

ACTIVATION_DICT = {
    Activation.identity: nn.Identity,
    Activation.relu: nn.ReLU,
    Activation.leaky_relu: nn.LeakyReLU,
    Activation.sigmoid: nn.Sigmoid,
    Activation.dmelu: layers.DmELU
}

dataset_norms = {
    dataset.cifar10: {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    dataset.kmnist: {'mean': [0.1307], 'std': [0.3081]},
    dataset.mnist: {'mean': [0.1307], 'std': [0.3081]},
    dataset.fmnist: {'mean': [0.2860], 'std': [0.3530]}
}


def path_from_config(config):
    return BASE_PATH / config['dataset'].value / (
        f"{config['mtype'].value}_{config['activation'].value}_{config['optimizer']}_"
        f"{config['nneurons']}_{config['nlayers']}_{str(config['lr'])[2:6]}"
    )

def load_configurations(args):
    """Load configurations either from CLI args or a YAML/JSON file."""
    if args.config_file:
        with open(args.config_file, 'r') as f:
            configs_raw = yaml.safe_load(f) if args.config_file.endswith('.yaml') else json.load(f)

        configs = []
        for c in configs_raw:

            keys, values = zip(*[
                (k, v if isinstance(v, list) else [v])
                for k, v in c.items()
            ])

            for combo in product(*values):
                cfg_dict = dict(zip(keys, combo))
                configs.append({
                    'mtype': ModelType[cfg_dict['mtype']],
                    'activation': Activation[cfg_dict['activation']],
                    'lr': float(cfg_dict['lr']),
                    'dataset': dataset[cfg_dict['dataset']],
                    'nneurons': int(cfg_dict['nneurons']),
                    'nlayers': int(cfg_dict['nlayers']),
                    'optimizer': cfg_dict.get('optimizer', 'adam'),
                    'reps': int(cfg_dict['reps']),
                })

        return configs

    elif args.model and args.dataset:
        return [{
            'mtype': ModelType[args.model],
            'activation': Activation[args.activation],
            'lr': args.lr,
            'dataset': dataset[args.dataset],
            'nneurons': args.nneurons,
            'nlayers': args.nlayers,
            'optimizer': args.optimizer
        }]

    else:
        return [
            {'mtype': x, 'activation': y, 'lr': z, 'dataset': w,
             'nneurons': nn, 'nlayers': nl, 'optimizer': o}
            for (x, y), z, w, nn, nl, o in product(
                [
                    (ModelType.euclidean, Activation.relu),
                    (ModelType.hyperbolic, Activation.relu),
                ],
                [1e-3],
                [dataset.mnist],
                [20],
                [1],
                ['sgd']
            )
        ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models (Docker-friendly)")
    parser.add_argument("--model", choices=[m.name for m in ModelType])
    parser.add_argument("--dataset", choices=[d.name for d in dataset])
    parser.add_argument("--activation", default="relu", choices=[a.name for a in Activation])
    parser.add_argument("--nneurons", type=int, default=20)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--config_file", type=str, help="Path to JSON/YAML configs.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()

    configs = load_configurations(args)

    for config in configs:

        path = path_from_config(config)
        if not path.exists():

            print(f"Working on {path}")
            path.mkdir(exist_ok=True, parents=True)

            TRANSFORM = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=dataset_norms[config['dataset']]['mean'],
                                                 std=dataset_norms[config['dataset']]['std']),
                torchvision.transforms.Lambda(lambda x: torch.flatten(x))
            ])


            dataset_class = DATASET_DICT[config['dataset']]
            train_dataset = dataset_class(root='files', train=True, transform=TRANSFORM, download=True)
            test_dataset = dataset_class(root='files', train=False, transform=TRANSFORM, download=True)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

            input_dim = next(iter(train_loader))[0].size()[1]
            output_dim = len(train_dataset.classes)
            nneurons = config['nneurons']

            model = MODEL_DICT[config['mtype']](
                input_dim,
                ACTIVATION_DICT[config['activation']],
                [nneurons] * config['nlayers'],
                head=output_dim
            ).to(device)

            for i in range(args.reps):
                seed = np.random.randint(0, 10000)
                train_model(model, config, train_loader, test_loader, path,
                            seed, epochs=args.epochs, save_model=args.save_model)
        else:
            print(f"--- {path} already exists ---")
