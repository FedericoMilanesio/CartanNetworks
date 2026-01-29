import torch
from torch.nn import init
from PGTS import HyperbolicAlgebra 
from geoopt import PoincareBall, Lorentz, ManifoldParameter, ManifoldTensor
from layers import HyperbolicLinear, HyperbolicActivation, HyperbolicRegressionLayer, HyperbolicEmbedding, HyperbolicConv2d, HyperLayer

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


class LogexpmapLinear(torch.nn.Module):
    def __init__(self, 
                size_in: int, 
                size_out: int, 
                bias: bool = True, 
                device=None, 
                dtype=None):
        super().__init__()
 
        self.instantiate_manifold()
    
        if size_out == 1:
            raise ValueError("size_out=1 is not supported")

        self.size_in, self.size_out = size_in, size_out
        
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weights = ManifoldParameter(
            torch.empty(size_out, size_in, **factory_kwargs)
        )

        if bias:
            self.bias = ManifoldParameter(
                ManifoldTensor(
                    torch.empty(size_in, **factory_kwargs), manifold = self.manifold
                    )
            )
        else:
            self.betas = None
            self.bias = None

        self.register_buffer('origin', self.manifold.origin(size_in, **factory_kwargs))
        
        self.reset_parameters()

    def instantiate_manifold(self):
        raise NotImplementedError

    def forward(
            self, 
            input: torch.Tensor
        ) -> torch.Tensor:

        transport = self.manifold.transp(self.origin, self.bias, input)
        point = self.manifold.expmap(self.bias, transport)
        tangent_vector = self.manifold.logmap(self.origin, point)
        return tangent_vector@self.weights.T

    
    def reset_parameters(self) -> None:

        init.kaiming_normal_(self.weights) 
        init.zeros_(self.bias)
        

class PoincareLinear(LogexpmapLinear):
    """Creates a Hyperbolic Linear Layer in Poincare coordinates
        
     Args:
        size_in (int): Number of input dimension (if the input is a torch.Tensor, size_int should be one higher)
        size_out (int): Number of output dimensions. Cannot be lower then 2 (use HyperbolicRegression instead).
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    """

    def instantiate_manifold(self):
        self.manifold = PoincareBall()

class LorentzLinear(LogexpmapLinear):
    """Creates a Hyperbolic Linear Layer in Lorentz coordinates
        
     Args:
        size_in (int): Number of input dimension (if the input is a torch.Tensor, size_int should be one higher)
        size_out (int): Number of output dimensions. Cannot be lower then 2 (use HyperbolicRegression instead).
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    """

    def instantiate_manifold(self):
        self.manifold = Lorentz()

    def reset_parameters(self):
        super().reset_parameters()
        self.bias = ManifoldParameter(self.manifold.projx(self.bias), manifold=self.manifold)
        
class PoincareNetwork(torch.nn.Module):
    def __init__(self, size, activation=torch.nn.Identity, layer_size_list=None, head=None):
        super().__init__()
        self.size = size
        if layer_size_list is None:
            layer_size_list = [16, 8, 4]
        self.hidden = torch.nn.Sequential(*[
            torch.nn.Sequential(
                PoincareLinear(s1, s2), activation()
            ) for s1, s2 in zip([size] + layer_size_list[:-1], layer_size_list)
        ])
        if head is None:
            self.head = torch.nn.Identity()
        else:
            self.head = head

    def forward(self, x):
        return self.head(self.hidden(x))
    
    def distance_from_euclidean(self, x):
        for _ in self.hidden:
            yield torch.tensor(0.)
    

class LorentzNetwork(torch.nn.Module):
    def __init__(self, size, activation=torch.nn.Identity, layer_size_list=None, head=None):
        super().__init__()
        self.size = size
        if layer_size_list is None:
            layer_size_list = [16, 8, 4]
        self.hidden = torch.nn.Sequential(*[
            torch.nn.Sequential(
                LorentzLinear(s1, s2), activation()
            ) for s1, s2 in zip([size] + layer_size_list[:-1], layer_size_list)
        ])
        if head is None:
            self.head = torch.nn.Identity()
        else:
            self.head = head

    def forward(self, x):
        return self.head(self.hidden(x))
    
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