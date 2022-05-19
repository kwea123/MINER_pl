import torch
from torch import nn
from einops import repeat


@torch.jit.script
def gaussian_activation(x, a):
    return torch.exp(-x**2/(2*a**2))


@torch.jit.script
def scaledsin_activation(x, a):
    return torch.sin(x*a)

# encoding matrices, used in PE
E_2d = torch.eye(2)

# defined in mipnerf360 https://arxiv.org/pdf/2111.12077.pdf page11
E_3d = torch.FloatTensor([
    0.8506508, 0, 0.5257311,
    0.809017, 0.5, 0.309017,
    0.5257311, 0.8506508, 0,
    1, 0, 0,
    0.809017, 0.5, -0.309017,
    0.8506508, 0, -0.5257311,
    0.309017, 0.809017, -0.5,
    0, 0.5257311, -0.8506508,
    0.5, 0.309017, -0.809017,
    0, 1, 0,
    -0.5257311, 0.8506508, 0,
    -0.309017, 0.809017, -0.5,
    0, 0.5257311, 0.8506508,
    -0.309017, 0.809017, 0.5,
    0.309017, 0.809017, 0.5,
    0.5, 0.309017, 0.809017,
    0.5, -0.309017, 0.809017,
    0, 0, 1,
    -0.5, 0.309017, 0.809017,
    -0.809017, 0.5, 0.309017,
    -0.809017, 0.5, -0.309017]).view(21, 3).T


class PE(nn.Module):
    """
    positional encoding
    """
    def __init__(self, P):
        """
        P: (d, F) encoding matrix
        """
        super().__init__()
        self.register_buffer("P", P)

    @property
    def out_dim(self):
        return self.P.shape[1]*2

    def forward(self, x):
        """
        x: (n_blocks, B, d)
        """
        x_ = x @ self.P # (n_blocks, B, F)
        return torch.cat([torch.sin(x_), torch.cos(x_)], -1) # (n_blocks, B, 2*F)


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
                ai = nn.Parameter(a*torch.ones(n_blocks, 1, 1))
            elif i < n_layers-1: # middle layers
                wi = nn.Parameter(torch.empty(n_blocks, n_hidden, n_hidden))
                bi = nn.Parameter(torch.empty(n_blocks, 1, n_hidden))
                ai = nn.Parameter(a*torch.ones(n_blocks, 1, 1))
            else: # last layer
                wi = nn.Parameter(torch.empty(n_blocks, n_hidden, n_out))
                bi = nn.Parameter(torch.empty(n_blocks, 1, n_out))
                if final_act == 'sigmoid':
                    ai = nn.Sigmoid()
                elif final_act == 'sin':
                    ai = nn.Parameter(a*torch.ones(n_blocks, 1, 1))

            # layer initialization
            if i == 0:
                nn.init.uniform_(wi, -1/(n_in**0.5), 1/(n_in**0.5))
                nn.init.uniform_(bi, -1/(n_in**0.5), 1/(n_in**0.5))
            else:
                nn.init.uniform_(wi, -1/(n_hidden**0.5), 1/(n_hidden**0.5))
                nn.init.uniform_(bi, -1/(n_hidden**0.5), 1/(n_hidden**0.5))

            setattr(self, f'w{i}', wi)
            setattr(self, f'b{i}', bi)
            setattr(self, f'a{i}', ai)

    def forward(self, x, b_chunks=16384, to_cpu=False, **kwargs):
        """
        Inputs:
            x: (n_blocks, B, n_in)
            b_chunks: int, @x is split into chunks of at most @b_chunks blocks

        Outputs:
            (n_blocks, B, n_out)
        """
        out = []
        for c in range(0, len(x), b_chunks):
            x_ = x[c:c+b_chunks]
            if 'pe' in kwargs: x_ = kwargs['pe'](x_)
            for i in range(self.n_layers):
                wi = getattr(self, f'w{i}')[c:c+b_chunks]
                bi = getattr(self, f'b{i}')[c:c+b_chunks]
                ai = getattr(self, f'a{i}')
                if i<self.n_layers-1:
                    x_ = gaussian_activation(x_@wi+bi, ai[c:c+b_chunks])
                else: # last layer
                    if self.final_act == 'sigmoid':
                        x_ = ai(x_@wi+bi)
                    elif self.final_act == 'sin':
                        x_ = scaledsin_activation(x_@wi+bi, ai[c:c+b_chunks])
            if to_cpu: x_ = x_.cpu()
            out += [x_]
        return torch.cat(out)
