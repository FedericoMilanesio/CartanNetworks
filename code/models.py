import torch
from torch.nn import init
from PGTS import HyperbolicAlgebra 
from geoopt import PoincareBall, Lorentz
from layers import HyperbolicLinear, HyperbolicActivation, HyperbolicRegressionLayer, HyperbolicEmbedding

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