from PGTS import HyperbolicAlgebra
import torch
from torch.nn import init
import functionals as hF
from geoopt import ManifoldParameter, ManifoldTensor, Sphere
from torch.nn.common_types import _size_1_t, _size_2_t
from typing import Union
import warnings
import math

class HyperbolicRegression(torch.nn.Module):
    """Creates a Hyperbolic Regression Layer
        
     Args:
        size_in (int): Number of input dimension.
        size_out (int): Number of output dimensions. Default to 1.

    """
    def __init__(self, 
                 size_in: int, 
                 size_out: int = 1,
                 device=None, 
                 dtype=None):

        super().__init__()

        self.size_in, self.size_out = size_in, size_out
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weights = torch.nn.Parameter(
            torch.empty((size_out, size_in-1), **factory_kwargs)
        )

        self.bias = torch.nn.Parameter(
            torch.empty((1, size_out), **factory_kwargs)
        )

        self.alpha = torch.nn.Parameter(
            torch.empty((1, size_out), **factory_kwargs)
        )

        self.reset_parameters()

    def forward(
            self, 
            input: torch.Tensor
            ) -> torch.Tensor:

        return hF.hregression(input, self.weights, self.alpha, self.bias)
    
    def reset_parameters(self) -> None:

        init.kaiming_normal_(self.weights)
        init.zeros_(self.bias)
        init.zeros_(self.alpha)

class HyperbolicEmbedding(torch.nn.Module):
    """Creates a Hyperbolic Embedding Layer, mapping the euclidean space into the hyperobolic space.
        
     Args:

    """
    def __init__(self, 
                 device=None, 
                 dtype=None):

        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

    def forward(
            self, 
            input: torch.Tensor
            ) -> torch.Tensor:

        f = input
        cartan = torch.zeros((*input.shape[:-1],1), dtype = f.dtype, device = f.device)

        return torch.cat([cartan, f], dim=-1)

class EuclideanProjection(torch.nn.Module):
    """Creates a Projection Layer, mapping the hyperbolic space into the euclidean space.
        
     Args:

    """
    def __init__(self, 
                 device=None, 
                 dtype=None):

        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

    def forward(
            self, 
            input: torch.Tensor
            ) -> torch.Tensor:

        fiber = input[...,1:]
        cartan = input[...,:1]

        return fiber * torch.exp(cartan)


class HyperbolicRegressionLayer:

    """Alias class of HyperbolicRegression"""

    def __new__(cls, *args, **kwargs):
        return HyperbolicRegression(*args, **kwargs)   

class HyperbolicLinear(torch.nn.Module):
    """Creates a Hyperbolic Linear Layer
        
     Args:
        size_in (int): Number of input dimension (if the input is a torch.Tensor, size_int should be one higher)
        size_out (int): Number of output dimensions. Cannot be lower then 2 (use HyperbolicRegression instead).
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    """
    def __init__(self, 
                 size_in: int, 
                 size_out: int, 
                 bias: bool = True, 
                 device=None, 
                 dtype=None):

        super().__init__()
        
        if size_out == 1:
            raise ValueError("size_out=1 is not supported, use 'torchyp.hnn.HyperbolicRegression' instead")

        self.size_in, self.size_out = size_in, size_out
        
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weights = ManifoldParameter(
            torch.empty(size_out-1, size_in-1, **factory_kwargs)
        )

        if bias:
            m = HyperbolicAlgebra()
            self.betas = ManifoldParameter(
                ManifoldTensor(
                    torch.empty(size_out, **factory_kwargs), manifold = m
                    )
            )
            self.bias = torch.nn.Parameter(
                torch.empty(size_out-1, **factory_kwargs)
            )
        else:
            self.betas = None
            self.bias = None
        


        s = Sphere()

        self.thetas = ManifoldParameter(
             ManifoldTensor(
                    torch.empty(size_out, **factory_kwargs), manifold = s
                    )
        )

        self.reset_parameters()


    def forward(
            self, 
            input: torch.Tensor
        ) -> torch.Tensor:

        return hF.hlinear(input, self.weights, self.thetas, self.bias, self.betas,)
    
    def reset_parameters(self) -> None:

        init.kaiming_normal_(self.weights) 
        init.zeros_(self.thetas)

        with torch.no_grad(): 
            self.thetas[0] = 1.0

        if self.betas is not None:
            init.zeros_(self.betas)
            init.zeros_(self.bias)

class HyperbolicConv1d(torch.nn.Module):

    """Applies a 1D convolution in hyperbolic space.
        
     Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Only groups=1 is implemented. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``

    """

    def __init__(self,  
                 in_channels:  int, 
                 out_channels: int, 
                 kernel_size: _size_1_t, 
                 stride: _size_1_t = 1, 
                 padding: Union[str, _size_1_t] = 0, 
                 dilation: _size_1_t = 1, 
                 bias: bool = True,
                 groups: int = 1,  
                 padding_mode: str = 'zeros', 
                 device=None, 
                 dtype=None):
        
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.bias_bool    = bias
        factory_kwargs    = {"device": device, "dtype": dtype}

        if groups != 1:
            warnings.warn(f"Number of groups different from one is not supported, setting groups=1")

        self.fiberConv1d = torch.nn.Conv1d(
                                           in_channels, 
                                           out_channels, 
                                           kernel_size, 
                                           stride=stride, 
                                           padding=padding, 
                                           dilation=dilation,
                                           bias=self.bias_bool, 
                                           padding_mode=padding_mode, 
                                           **factory_kwargs)
        
        if self.bias_bool:           

            m = HyperbolicAlgebra()
            self.betas = ManifoldParameter(
                ManifoldTensor(
                    torch.empty(out_channels + 1, **factory_kwargs), manifold = m
                    )
            )
        else:
            self.betas = None
        
        s = Sphere()

        self.thetas = ManifoldParameter(
             ManifoldTensor(
                    torch.empty(out_channels + 1, **factory_kwargs), manifold = s
                    )
        )

        self.reset_parameters()

    def forward(self, 
                x: torch.Tensor
            ) -> torch.Tensor:

        m = HyperbolicAlgebra()

        fiber_reshape = m.fiber(x).reshape(x.size(0), 
                                              self.in_channels, 
                                              (x.size(1)-1)//self.in_channels)
        fiber = self.fiberConv1d(fiber_reshape)
        cartan = m.cartan(x)[..., None]

        flattened_fiber = fiber.view(fiber.size(0), -1)
        flattened_cartan = cartan.view(fiber.size(0), -1)

        h = m.from_cartan_and_fiber(flattened_cartan, flattened_fiber)
        n = fiber.size(2)

        if self.bias_bool:
            bias_stacked = torch.cat((self.betas[:1], 
                                      self.betas[1:].repeat_interleave(n)/math.sqrt(n)))
            m.group_mul(h, bias_stacked)

        theta_stacked = torch.cat((self.thetas[:1], 
                                   self.thetas[1:].repeat_interleave(n)/math.sqrt(n)))

        return m.fiber_rotation(h, theta_stacked)
    
    def reset_parameters(self) -> None:
        
        if self.bias_bool:
            init.zeros_(self.betas)

        init.zeros_(self.thetas)

        with torch.no_grad(): 
            self.thetas[0] = 1.0

        self.fiberConv1d.reset_parameters()
  

class HyperbolicConv2d(torch.nn.Module):

    """Applies a 2D convolution in hyperbolic space.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Only groups=1 is implemented. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``

    
    """

    def __init__(self,  
                 in_channels:  int, 
                 out_channels: int, 
                 kernel_size: _size_2_t, 
                 stride: _size_2_t = 1, 
                 padding: Union[str, _size_2_t] = 0, 
                 dilation: _size_2_t = 1, 
                 groups: int = 1,
                 bias: bool= True,  
                 padding_mode='zeros', 
                 device=None, 
                 dtype=None):
        
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.bias_bool    = bias
        factory_kwargs = {"device": device, "dtype": dtype}

        if groups is not None:
            warnings.warn(f"Number of groups different from one is not supported, setting groups=1")
 

        self.fiberConv2d = torch.nn.Conv2d(in_channels, 
                                           out_channels, 
                                           kernel_size, 
                                           stride=stride, 
                                           padding=padding, 
                                           dilation=dilation,
                                           bias=self.bias_bool, 
                                           padding_mode=padding_mode, 
                                           **factory_kwargs)
        
        if self.bias_bool:           

            m = HyperbolicAlgebra()
            self.betas = ManifoldParameter(
                ManifoldTensor(
                    torch.empty(out_channels + 1, **factory_kwargs), manifold = m
                    )
            )
        else:
            self.betas = None
        
        s = Sphere()

        self.thetas = ManifoldParameter(
             ManifoldTensor(
                    torch.empty(out_channels + 1, **factory_kwargs), manifold = s
                    )
        )

        self.reset_parameters()

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        side = math.isqrt((x.size(1)-1)//self.in_channels)
        m = HyperbolicAlgebra()
            
        if side**2 * self.in_channels + 1 != x.size(1):
            raise ValueError("Non-square inputs are not supported (yet)")

        fiber_reshape = m.fiber(x).reshape(x.size(0), self.in_channels, side, side)
        fiber = self.fiberConv2d(fiber_reshape)
        cartan = m.cartan(x)[..., None]

        flattened_fiber = fiber.view(fiber.size(0), -1)
        flattened_cartan = cartan.view(fiber.size(0), -1)

        h = m.from_cartan_and_fiber(flattened_cartan, flattened_fiber)
        n = fiber.size(2)*fiber.size(3)

        if self.bias_bool:

            bias_stacked = torch.cat((self.betas[:1], self.betas[1:].repeat_interleave(n)*math.sqrt(n)))
            m.group_mul(h, bias_stacked)

        theta_stacked = torch.cat((self.thetas[:1], self.thetas[1:].repeat_interleave(n)*math.sqrt(n)))

        return m.fiber_rotation(h, theta_stacked)
    
    def reset_parameters(self) -> None:
        
        if self.bias_bool:  
            init.zeros_(self.betas)
        init.zeros_(self.thetas)

        with torch.no_grad(): 
            self.thetas[0] = 1.0

        self.fiberConv2d.reset_parameters()

class HyperbolicActivation(torch.nn.Module):  
    """Custom hyperbolic activations"""
    def __init__(self, 
                 activation):
        super().__init__()
        self.activation = activation()
        self._name = "h-" + activation.__name__
    
    @property
    def __name__(self):
        return self._name
    
    def forward(self, x) -> torch.Tensor:
        m = HyperbolicAlgebra()
        return torch.Tensor(torch.cat([m.cartan(x), self.activation(m.fiber(x))], dim = -1))
    
class DmELU(torch.nn.Module):
    def __init__(self, alpha = 0.1):
       super().__init__()
       self.alpha = alpha
       
    def forward(self, input):
      return (torch.nn.functional.elu(input) + self.alpha * input) / (1. + self.alpha)
    
    def __name__(self):
        return "DmELU"
    

