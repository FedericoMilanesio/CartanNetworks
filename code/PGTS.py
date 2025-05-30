import torch
from geoopt import Manifold

class HyperbolicAlgebra(Manifold):
    name = "HyperbolicAlgebra"

    def __init__(self, ndim=1):
        super().__init__()
        self.ndim = ndim
        self.eps = 1e-8
    
    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5):
        """Checks if a tensor is on the manifold.

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            atol (_type_, optional): tolerance. Defaults to 1e-5.

        Returns:
            tuple:
                - bool: True if validation succeeds, False otherwise.
                - str or None: An error message if validation fails, otherwise None.

        """        
        return True, None

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5):
        """Checks if a tangent tensor 'u' is on the tangent space of 'x'.

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            u (torch.Tensor): tensor of the tangent space of x.
            atol (_type_, optional): tolerance. Defaults to 1e-5.
        
        Returns:
            tuple:
                - bool: True if validation succeeds, False otherwise.
                - str or None: An error message if validation fails, otherwise None.

        """        
        return True, None
        
    def from_cartan_and_fiber(self, cartan: torch.Tensor, fiber: torch.Tensor):
        """Returns a tensor on the manifold from its cartan and fiber components

        Args:
            cartan (torch.Tensor): cartan component of a tensor on the hyperbolic manifold.
            fiber (torch.Tensor): fiber component of a tensor on the hyperbolic manifold.

        Returns:
            torch.Tensor: tensor on the hyperbolic manifold.
        """        
        return torch.cat([cartan, fiber], dim=-1)

    def egrad2rgrad(self, x: torch.Tensor, grad: torch.Tensor):
        """Returns the Riemannian gradient in point x from the Euclidean gradient

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            grad (torch.Tensor): Euclidean gradient.

        Returns:
            torch.Tensor: Riemannian gradient in point x.
        """        
        dotg = (self.fiber(x) * grad[..., 1:]).sum(dim=-1) 
        g0_new = grad[..., 0] - dotg
        g1_new = grad[..., 1:] - g0_new * self.fiber(x)
        return torch.cat((g0_new.unsqueeze(0), g1_new))
    
    def cartan(self, x: torch.Tensor) -> torch.Tensor:
        """Return the Cartan component of the HyperbolicAlgebra tensor

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.

        Returns:
            torch.Tensor: cartan component of a tensor on the hyperbolic manifold.
        """
        if len(x.shape) == 0:
            return self
        
        cartan = x[...,:1]
        return cartan
    
    def fiber(self, x: torch.Tensor) -> torch.Tensor:
        """Return the fiber component of the HyperbolicAlgebra tensor

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.

        Returns:
            torch.Tensor: fiber component of a tensor on the hyperbolic manifold.
        """
        if len(x.shape) == 0:
            return self
        
        fiber = x[..., 1:]
        return fiber

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False):
        """Inner product of tangent tensor u and v in point x. If v is None, defaults to u*u at point x

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            u (torch.Tensor): tensor of the tangent space of x.
            v (torch.Tensor, optional): tensor of the tangent space of x.. Defaults to None.
            keepdim (bool, optional): whether to keep the dimension of the input. Defaults to False.

        Returns:
            torch.Tensor: inner product of tangent tensor u and v in point x. If v is None, defaults to u*u at point x
        """
        if v is None:
            inner = ((u[...,0]*self.fiber(x) + u[...,1:])**2).sum(-1, keepdim=keepdim) + u[...,0]**2
        else:
            inner = ((u[...,0]*self.fiber(x) + u[...,1:])*(v[...,0]*self.fiber(x) + v[...,1:])).sum(-1, keepdim=keepdim) + u[...,0]*v[...,0]

        return inner
    
    def proju(self, x: torch.Tensor, u: torch.Tensor):
        """Projection tangent tensor u to the tangent space to x.

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            u (torch.Tensor): tensor of the tangent space of x.

        Returns:
            torch.Tensor: projection of u onto tangent space of x.
        """        
        return u
    
    def projx(self, x: torch.Tensor):
        """Projection of tensor x onto the manifold.

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.

        Returns:
            torch.Tensor: projection of x onto the manifold.
        """      
        return x
    
    def retr(self, x: torch.Tensor, u: torch.Tensor):
        """Retraction of tangent tensor u from point on the manifold x.

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            u (torch.Tensor): tensor of the tangent space of x.

        Returns:
            torch.Tensor: retraction of tangent tensor u from point on the manifold x
        """        
        return x + u
    
    def dist(self, x: torch.Tensor, y: torch.Tensor):
        """Computes the distance between points x and y

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            y (torch.Tensor): tensor on the hyperbolic manifold.

        Returns:
            torch.Tensor: distance between x and y.
        """        

        dxy = self.group_mulinv(x, y)
        fiber_sum = torch.sum(dxy[...,1:]**2, dim=-1)
        cartan_exp = torch.exp(dxy[...,0]).squeeze()
        norm =  torch.acosh(self.eps + 
                            (0.5 * ( 1 / cartan_exp + 
                                cartan_exp * (fiber_sum + 1)
                            )))
        return norm

    
    def group_mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Group operation between x and y (y*x)

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            y (torch.Tensor): tensor on the hyperbolic manifold.

        Returns:
            torch.Tensor: group operation y*x
        """
        cartan = self.cartan(x) + self.cartan(y)
        fiber =self.fiber(x) + torch.exp(-self.cartan(x) ) * self.fiber(y)
        return self.from_cartan_and_fiber(cartan, fiber)
    
    def group_mulinv(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Group operation between x and y^-1 (y^-1*x)

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            y (torch.Tensor): tensor on the hyperbolic manifold.

        Returns:
            torch.Tensor: group operation (y^-1*x)
        """
        cartan = self.cartan(x) - self.cartan(y)
        fiber = self.fiber(x) - torch.exp(-cartan) * self.fiber(y)
        return self.from_cartan_and_fiber(cartan, fiber)

    def group_inv(self, x: torch.Tensor) -> torch.Tensor:
        """Group inversion of x

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.

        Returns:
            torch.Tensor: x^-1
        """
        return self.from_cartan_and_fiber(-self.cartan(x), -torch.exp(self.cartan(x)) * self.fiber(x))

    def log0(self, x: torch.Tensor) -> torch.Tensor:
        """Riemannian logarithmic map in the origin

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.

        Returns:
            torch.Tensor: log_0(x)
        """
        N_x = self.dist(x, self.zero(x))
        denom = torch.sinh(N_x) + self.eps
        cartan = torch.cosh(N_x) - torch.exp(-self.cartan(x))
        return N_x / denom * torch.cat([cartan, self.fiber(x)], dim=-1)
    
    def exp0(self, v: torch.Tensor, *, dim=-1):
        """Riemannian exponential map in the origin

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.

        Returns:
            torch.Tensor: exp_0(x)
        """
        u = torch.norm(v, dim = -1)
        cartan = -torch.log(-torch.sinh(u) / u * v[..., :1] + torch.cosh(u))
        fiber =  torch.sinh(u) / u * v[..., 1:]
        return self.from_cartan_and_fiber(cartan, fiber)


    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Tranport of tangent tensor v from x to y.

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            y (torch.Tensor): tensor on the hyperbolic manifold.
            v (torch.Tensor): tensor of the tangent space of x.

        Returns:
            torch.Tensor: tensor of the tangent space of y.
        """
        v[..., :1] += (v[...,1:] * (self.fiber(y) - self.fiber(x))).sum(dim=-1)
        return v
    
    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=True):
        """Norm of tensor u in tangent space of x

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            u (torch.Tensor): tensor of the tangent space of x.
            keepdim (bool, optional): whether to keep dimensions. Defaults to True.

        Returns:
            torch.Tensor: norm of u.
        """
        norm = ((u[...,0]*self.fiber(x) + u[...,1:])**2).sum(-1, keepdim=keepdim) + u[...,0]**2
        return torch.sqrt(norm)
    
    def expmap(self, x: torch.Tensor, v: torch.Tensor, *, dim=-1):
        """Riemannian exponential map of tangent vector v in point x (exp_x(v)).

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            v (torch.Tensor): tensor of the tangent space of x.
        Returns:
            torch.Tensor: exp_x(v)
        """
        v[..., :1] -= (v[...,1:]*self.fiber(x)).sum(dim=-1)
        z = self.exp0(v)
        return self.group_mul(z, x)

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1):
        """Riemannian logarithmic map of point y in point x (exp_x(y)).

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            y (torch.Tensor): tensor on the hyperbolic manifold.
        Returns:
            torch.Tensor: exp_x(y)
        """
        x0, y0 = self.cartan(x), self.cartan(y)
        x_fiber = self.fiber(x)
        y_fiber = self.fiber(y)
        
        U = torch.exp(-x0 + y0)
        V = 1. / U
        
        diff = x_fiber - y_fiber * U.unsqueeze(-1) 
        Q = torch.sum(diff**2, dim=dim)
        
        A = 0.5 * (U + V * (1 + Q))
        
        S = torch.sum(y_fiber * diff, dim=dim)
        grad_A_x0 = A - U + V * U * S
        
        grad_A_xfiber = V.unsqueeze(-1) * diff
        
        distance = torch.acosh(1e-8 + A)
        denom = torch.sinh(distance)

        grad_d_x0 = grad_A_x0 / denom
        grad_d_xfiber = grad_A_xfiber / denom.unsqueeze(-1)
        
        
        logmap_x0 = -distance * grad_d_x0
        logmap_xfiber = -distance.unsqueeze(-1) * grad_d_xfiber
        
        logmap_val = torch.cat([logmap_x0.unsqueeze(-1), logmap_xfiber], dim=-1)
        return logmap_val

    def zero(self, x: torch.Tensor):
        """Creates a zero-like tensor from a given tensor.

        Args:
            x (torch.Tensor): point in the manifold

        Returns:
            torch.Tensor: zeros.
        """        
        return torch.zeros_like(x)
    
    def fiber_rotation(self, x: torch.Tensor,  phi: torch.Tensor):
        """Fiber rotation of point x by phi element with spherical coordinates

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
            phi (torch.Tensor): tensor with spherical coordinates.

        Returns:
            torch.Tensor: fiber rotation of x by phi
        """    
        m = self
        mod = (m.fiber(x)**2).sum(axis = -1, keepdim = True)
        c_exp = torch.exp(m.cartan(x))

        A = (mod+1)*c_exp
        B = 1 / c_exp

        if phi.size()[-1] - 1 != m.fiber(x).size()[-1]:
            raise ValueError(f"The number of angles must match the length of the fiber: {phi.size()[-1]} != {m.fiber(x).size()[-1]}")
        
        dot_prod = (m.fiber(x) @ phi[1:])[...,None]
        yc = -torch.log(0.5*(-phi[0]*(A - B) + (B + A)) - dot_prod + self.eps)

        if torch.abs(phi[0] + 1) < self.eps:
            yl = -m.fiber(x)
        else:
            yl = m.fiber(x) + phi[1:] * (-dot_prod / (phi[0] + 1 + self.eps) + 0.5 * (B - A))

        return self.from_cartan_and_fiber(yc, yl)


    def to_ball_representation(self, x: torch.Tensor):
        """Computes the coordinates of the Poincarè ball representation of the element.

        Args:
            x (torch.Tensor): tensor on the hyperbolic manifold.
        Returns:
            torch.Tensor: tensor in Poincarè ball coordinates.
        """
        cartan = self.cartan(x)
        fiber_sum = torch.sum(self.fiber(x)**2, dim=-1)[...,None]
        cartan_exp = torch.exp(cartan)

        denom = ((cartan_exp**2) * (fiber_sum + 1) + 2 * cartan_exp + 1)

        x_1 = -((1 - (cartan_exp**2) * (fiber_sum + 1))/denom)
        x_fiber = 2 * cartan_exp * self.fiber(x)/denom
        
        return torch.cat([x_1, x_fiber], dim=-1)
