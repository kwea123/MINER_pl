import torch
from torch import nn


@torch.jit.script
def gaussian_activation(x, a):
    return torch.exp(-x**2/(2*a**2))


@torch.jit.script
def scaledsin_activation(x, a):
    return torch.sin(x*a)


class BlockMLP(nn.Module):
    """
    A BlockMLP consists of all the MLPs of a certain scale.
    All MLPs are inferred at the same time without using for loop.
    """
    def __init__(self, n_blocks, n_in, n_out,
                 n_layers, n_hidden, final_act,
                 a=0.1):
        super().__init__()

        self.n_layers = n_layers
        self.final_act = final_act
        for i in range(n_layers):
            if i == 0: # first layer
                wi = nn.Parameter(torch.empty(n_blocks, n_in, n_hidden))
                bi = nn.Parameter(torch.empty(n_blocks, 1, n_hidden))
                setattr(self, f'a{i}', nn.Parameter(a*torch.ones(n_blocks, 1, 1)))
            elif i < n_layers-1: # middle layers
                wi = nn.Parameter(torch.empty(n_blocks, n_hidden, n_hidden))
                bi = nn.Parameter(torch.empty(n_blocks, 1, n_hidden))
                setattr(self, f'a{i}', nn.Parameter(a*torch.ones(n_blocks, 1, 1)))
            else: # last layer
                wi = nn.Parameter(torch.empty(n_blocks, n_hidden, n_out))
                bi = nn.Parameter(torch.empty(n_blocks, 1, n_out))
                if final_act == 'sigmoid':
                    setattr(self, f'a{i}', nn.Sigmoid())
                elif final_act == 'sin':
                    setattr(self, f'a{i}', nn.Parameter(a*torch.ones(n_blocks, 1, 1)))

            # layer initialization
            if i == 0:
                nn.init.uniform_(wi, -1/(n_in**0.5), 1/(n_in**0.5))
                nn.init.uniform_(bi, -1/(n_in**0.5), 1/(n_in**0.5))
            else:
                nn.init.uniform_(wi, -1/(n_hidden**0.5), 1/(n_hidden**0.5))
                nn.init.uniform_(bi, -1/(n_hidden**0.5), 1/(n_hidden**0.5))

            self.register_parameter(f'w{i}', wi)
            self.register_parameter(f'b{i}', bi)

    def forward(self, x):
        """
        Inputs:
            x: (n_blocks, B, n_in)
        
        Outputs:
            (n_blocks, B, n_out)
        """
        for i in range(self.n_layers):
            wi = getattr(self, f'w{i}')
            bi = getattr(self, f'b{i}')
            ai = getattr(self, f'a{i}')
            if i<self.n_layers-1:
                x = gaussian_activation(x@wi+bi, ai)
            else: # last layer
                if self.final_act == 'sigmoid':
                    x = ai(x@wi+bi)
                elif self.final_act == 'sin':
                    x = scaledsin_activation(x@wi+bi, ai)
        return x
