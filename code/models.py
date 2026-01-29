import torch
from torch.nn import init
from PGTS import HyperbolicAlgebra 
from geoopt import PoincareBall, Lorentz, ManifoldParameter, ManifoldTensor
from layers import HyperbolicLinear, HyperbolicActivation, HyperbolicRegressionLayer, HyperbolicEmbedding, HyperbolicConv2d, HyperLayer

from lorentz_fully.lorentz.layers.LFC import LorentzFullyConnected
from lorentz_fully.lorentz.layers.LModules import LorentzAct
from lorentz_fully.lorentz.layers.LMLR import LorentzMLR
from lorentz_fully.lorentz.manifold import CustomLorentz

from hyptorch.nn.modules.linear import HypLinear
from hyptorch.nn.modules.mlr import HyperbolicMLR
from hyptorch.manifolds.poincare_ball import PoincareBall
from hyptorch.nn.modules.manifold import ToPoincare

from h_plusplus.modules.linear import PoincareLinear
from h_plusplus.modules.multinomial_logistic_regression import UnidirectionalPoincareMLR
from h_plusplus.manifolds.stereographic.manifold import PoincareBall as BallPLusPlus

class HyperbolicNetwork(torch.nn.Module):
    def __init__(self, size, activation=torch.nn.Identity, layer_size_list=None, head=None):
        super().__init__()
        self.size = size
        if layer_size_list is None:
            layer_size_list = [16, 8, 4]
        self.hidden = torch.nn.Sequential(HyperbolicEmbedding(self.size), *[
            torch.nn.Sequential(
                HyperbolicLinear(s1, s2), HyperbolicActivation(activation)
            ) for s1, s2 in zip([size+1] + layer_size_list[:-1], layer_size_list)
        ])
        if head is None:
            self.head = HyperbolicRegressionLayer(layer_size_list[-1])
        else:
            self.head = head

    def forward(self, x):
        return self.head(self.hidden(x))
    
    def distance_from_euclidean(self, x):
        repr = x
        m = HyperbolicAlgebra()
        for h in self.hidden:
          repr = h(repr)
          yield torch.mean(m.cartan(repr).pow(2))

    def fibernorm(self, x):
        repr = x
        m = HyperbolicAlgebra()
        for h in self.hidden:
          repr = h(repr)
          yield torch.mean(m.fiber(repr).pow(2).sum(dim=-1))

class PoincareNetwork(torch.nn.Module):
    def __init__(self, size, activation=torch.nn.Identity, layer_size_list=None, head=None):
        super().__init__()

        self.size = size

        if layer_size_list is None:
            layer_size_list = [16, 8, 4]

        self.manifold = PoincareBall(curvature=1, trainable_curvature=False)
        self.embed = ToPoincare(self.manifold)

        self.hidden = torch.nn.Sequential(*[
            torch.nn.Sequential(
                HypLinear(s1, s2, manifold=self.manifold), activation()
            ) for s1, s2 in zip([size] + layer_size_list[:-1], layer_size_list)
        ])
        if head is None:
            self.head = HyperbolicMLR(ball_dim=layer_size_list[-1], n_classes=1, manifold=self.manifold)
        else:
            self.head = HyperbolicMLR(ball_dim=layer_size_list[-1], n_classes=head, manifold=self.manifold)

    def forward(self, x):
        return self.head(self.hidden(self.embed(x)))

class PlusPlusNetwork(torch.nn.Module):
    def __init__(self, size, activation=torch.nn.Identity, layer_size_list=None, head=None):
        super().__init__()

        self.size = size

        if layer_size_list is None:
            layer_size_list = [16, 8, 4]

        self.manifold = BallPLusPlus(c=1)
        self.embed = self.manifold.expmap0

        self.hidden = torch.nn.Sequential(*[
            torch.nn.Sequential(
                PoincareLinear(s1, s2, ball = self.manifold), activation()
            ) for s1, s2 in zip([size] + layer_size_list[:-1], layer_size_list)
        ])
        if head is None:
            self.head = UnidirectionalPoincareMLR(layer_size_list[-1], 1, ball = self.manifold)
        else:
            self.head = UnidirectionalPoincareMLR(layer_size_list[-1], head, ball = self.manifold)

    def forward(self, x):
        return self.head(self.hidden(self.embed(x)))
    

class LorentzNetwork(torch.nn.Module):
    def __init__(self, size, activation=torch.nn.Identity, layer_size_list=None, head=None):
        super().__init__()
        self.size = size
        if layer_size_list is None:
            layer_size_list = [16, 8, 4]
        self.manifold = CustomLorentz()
        self.hidden = torch.nn.Sequential(*[
            torch.nn.Sequential(
                LorentzFullyConnected(s1, s2), LorentzAct(activation())
            ) for s1, s2 in zip([size] + layer_size_list[:-1], layer_size_list)
        ])
        if head is None:
            self.head = LorentzMLR(layer_size_list[-1], 1)
        else:
            self.head = LorentzMLR(layer_size_list[-1], head)

    def forward(self, x):
        return self.head(self.hidden(self.manifold.projx(x)))
    
class EuclideanNetwork(torch.nn.Module):
    def __init__(self, size, activation=torch.nn.Identity, layer_size_list=None, head=None):
        super().__init__()
        self.size = size
        if layer_size_list is None:
            layer_size_list = [17, 5, 3]
        self.hidden = torch.nn.Sequential(*[
            torch.nn.Sequential(
                torch.nn.Linear(s1, s2), activation()
            ) for s1, s2 in zip([size] + layer_size_list[:-1], layer_size_list)
        ])
        if head is None:
            self.head=torch.nn.Linear(layer_size_list[-1], 1)
        else:
            self.head = head
      
    def forward(self, x):
        return self.head(self.hidden(x))

class HyperAlexNetCifar(torch.nn.Module):
    def __init__(self, activation=torch.nn.ReLU, num_classes=100, input_channels=3, *args, **kwargs):
        super().__init__()
        self.features = torch.nn.Sequential(
            HyperbolicEmbedding(),
            # First convolutional layer: 3 -> 96, kernel size=11, stride=4, padding=2
            HyperbolicConv2d(input_channels, 96, kernel_size=11, stride=4, padding=2),  # Output: (96, 55, 55)
            HyperbolicActivation(activation),
            # Local Response Norm (LRN) after first activation
            HyperLayer(torch.nn.LocalResponseNorm, in_shape=(96, 55, 55), size=5, alpha=0.0001, beta=0.75, k=2),
            HyperLayer(torch.nn.MaxPool2d, in_shape=(96, 55, 55), kernel_size=3, stride=2),  # Output: (96, 27, 27)

            # Second convolutional layer: 96 -> 256, kernel size=5, padding=2
            HyperbolicConv2d(96, 256, kernel_size=5, padding=2),  # Output: (256, 27, 27)
            HyperbolicActivation(activation),
            # LRN after second activation
            HyperLayer(torch.nn.LocalResponseNorm, in_shape=(256, 27, 27), size=5, alpha=0.0001, beta=0.75, k=2),
            HyperLayer(torch.nn.MaxPool2d, in_shape=(256, 27, 27), kernel_size=3, stride=2),  # Output: (256, 13, 13)

            # Third convolutional layer: 256 -> 384, kernel size=3, padding=1
            HyperbolicConv2d(256, 384, kernel_size=3, padding=1),  # Output: (384, 13, 13)
            HyperbolicActivation(activation),

            # Fourth convolutional layer: 384 -> 384, kernel size=3, padding=1
            HyperbolicConv2d(384, 384, kernel_size=3, padding=1),  # Output: (384, 13, 13)
            HyperbolicActivation(activation),

            # Fifth convolutional layer: 384 -> 256, kernel size=3, padding=1
            HyperbolicConv2d(384, 256, kernel_size=3, padding=1),  # Output: (256, 13, 13)
            HyperbolicActivation(activation),
            HyperLayer(torch.nn.MaxPool2d, in_shape=(256, 13, 13), kernel_size=3, stride=2),  # Output: (256, 6, 6)
        )
        self.classifier = torch.nn.Sequential(
            HyperLayer(torch.nn.Dropout, p=0.5),
            # First fully connected layer: 256 * 6 * 6 -> 4096
            HyperbolicLinear(256 * 6 * 6 + 1, 4096),
            HyperbolicActivation(activation),
            HyperLayer(torch.nn.Dropout, p=0.5),
            # Second fully connected layer: 4096 -> 4096
            HyperbolicLinear(4096, 4096),
            HyperbolicActivation(activation),
            # Final output layer: 4096 -> 100 (for CIFAR-100)
            HyperbolicRegressionLayer(4096, num_classes),
        )

    def forward(self, data):
        # Flatten input for hyperbolic layers
        x = self.features(data.flatten(start_dim=1))
        # Flatten again before passing to the classifier
        x = x.flatten(start_dim=1)
        return self.classifier(x)

class AlexNetCifar(torch.nn.Module):
    def __init__(self, activation=torch.nn.ReLU, num_classes=100, input_channels=3, *args, **kwargs):
        super().__init__()
        self.features = torch.nn.Sequential(
            # First convolutional layer: 3 -> 96, kernel size=11, stride=4, padding=2
            torch.nn.Conv2d(input_channels, 96, kernel_size=11, stride=4, padding=2),  # Output: (96, 55, 55)
            activation(inplace=True),
            # Local Response Norm (LRN) after first activation
            torch.nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),  # Output: (96, 27, 27)

            # Second convolutional layer: 96 -> 256, kernel size=5, padding=2
            torch.nn.Conv2d(96, 256, kernel_size=5, padding=2),  # Output: (256, 27, 27)
            activation(inplace=True),
            # LRN after second activation
            torch.nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),  # Output: (256, 13, 13)

            # Third convolutional layer: 256 -> 384, kernel size=3, padding=1
            torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),  # Output: (384, 13, 13)
            activation(inplace=True),

            # Fourth convolutional layer: 384 -> 384, kernel size=3, padding=1
            torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),  # Output: (384, 13, 13)
            activation(inplace=True),

            # Fifth convolutional layer: 384 -> 256, kernel size=3, padding=1
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Output: (256, 13, 13)
            activation(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),  # Output: (256, 6, 6)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            # First fully connected layer: 256 * 6 * 6 -> 4096
            torch.nn.Linear(256 * 6 * 6, 4096),
            activation(inplace=True),
            torch.nn.Dropout(p=0.5),
            # Second fully connected layer: 4096 -> 4096
            torch.nn.Linear(4096, 4096),
            activation(inplace=True),
            # Final output layer: 4096 -> num_classes
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, data):
        # Pass through convolutional layers
        x = self.features(data)
        # Flatten before passing to the classifier
        x = x.flatten(start_dim=1)
        return self.classifier(x)