import torch
from PGTS import HyperbolicAlgebra

def hregression(x: torch.Tensor, 
                weights: torch.Tensor, 
                alpha=None,
                bias=None, 
                ):
    
    m = HyperbolicAlgebra()

    cart_exp = torch.exp(m.cartan(x))
    fiber_mod = (m.fiber(x)**2).sum(axis = -1)[:,None]
    res = (m.fiber(x) @ weights.t())

    if alpha is not None:
        res += (1 + fiber_mod) * cart_exp * alpha

    if bias is not None:
        res += 1 / cart_exp * bias

    return res

def hlinear(x: torch.Tensor, 
            weights: torch.Tensor, 
            thetas: torch.Tensor, 
            bias=None,
            betas=None):

    m = HyperbolicAlgebra()
    f = m.fiber(x) @ weights.t() 
    
    if bias is not None:
        f = f + bias
        
    cartan = m.cartan(x)
   
    h = m.from_cartan_and_fiber(cartan, f)
        
    if betas is not None:
        h = m.group_mul(h, betas)

    return m.fiber_rotation(h, thetas)

class DmELU(torch.nn.Module):
    def __init__(self, alpha = 0.1):
       super().__init__()
       self.alpha = alpha
       
    def forward(self, input):
      return (torch.nn.functional.elu(input) + self.alpha * input) / (1. + self.alpha)
    
    def __name__(self):
        return "DmELU"
