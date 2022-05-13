import torch
from torch import nn


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
                setattr(self, f'a{i}', GaussianActivation(n_blocks, a))
            elif i < n_layers-1: # middle layers
                wi = nn.Parameter(torch.empty(n_blocks, n_hidden, n_hidden))
                bi = nn.Parameter(torch.empty(n_blocks, 1, n_hidden))
                setattr(self, f'a{i}', GaussianActivation(n_blocks, a))
            else: # last layer
                wi = nn.Parameter(torch.empty(n_blocks, n_hidden, n_out))
                bi = nn.Parameter(torch.empty(n_blocks, 1, n_out))
                if final_act == 'sigmoid':
                    setattr(self, f'a{i}', nn.Sigmoid())
                elif final_act == 'sin':
                    setattr(self, f'a{i}', ScaledSinActivation(n_blocks, a))

            # layer initialization
            if i == 0:
                nn.init.uniform_(wi, -1/(n_in**0.5), 1/(n_in**0.5))
                nn.init.uniform_(bi, -1/(n_in**0.5), 1/(n_in**0.5))
            else:
                nn.init.uniform_(wi, -1/(n_hidden**0.5), 1/(n_hidden**0.5))
                nn.init.uniform_(bi, -1/(n_hidden**0.5), 1/(n_hidden**0.5))

            self.register_parameter(f'w{i}', wi)
            self.register_parameter(f'b{i}', bi)

    def forward(self, x, training_blocks, b_chunk):
        """
        Inputs:
            x: (n_blocks, B, n_in)
            training_blocks: None (=all blocks) or bool of shape (n_blocks)
            b_chunk: int, split @x into at most this number of blocks
        
        Outputs:
            (n_training_blocks, B, n_out)
        """
        if training_blocks is None: # called in val only
            out = []
            for c in range(0, len(x), b_chunk):
                x_ = x[c:c+b_chunk]
                for i in range(self.n_layers):
                    wi = getattr(self, f'w{i}')[c:c+b_chunk]
                    bi = getattr(self, f'b{i}')[c:c+b_chunk]
                    ai = getattr(self, f'a{i}')
                    if i<self.n_layers-1 or self.final_act=='sin':
                        x_ = ai(x_@wi+bi, training_blocks, c, c+b_chunk)
                    else:
                        x_ = ai(x_@wi+bi)
                out += [x_]
            out = torch.cat(out, 0)
            return out

        x = x[training_blocks]
        out = []
        for c in range(0, len(x), b_chunk):
            x_ = x[c:c+b_chunk]
            for i in range(self.n_layers):
                wi = getattr(self, f'w{i}')[training_blocks][c:c+b_chunk]
                bi = getattr(self, f'b{i}')[training_blocks][c:c+b_chunk]
                ai = getattr(self, f'a{i}')
                if i<self.n_layers-1 or self.final_act=='sin':
                    x_ = ai(x_@wi+bi, training_blocks, c, c+b_chunk)
                else:
                    x_ = ai(x_@wi+bi)
            out += [x_]
        out = torch.cat(out, 0)
        return out


class ScaledSinActivation(nn.Module):
    def __init__(self, n_blocks, a=1):
        super().__init__()
        self.register_parameter('a',
            nn.Parameter(a*torch.ones(n_blocks, 1, 1)))
    
    def forward(self, x, training_blocks, chunk_s, chunk_e):
        """
        Inputs:
            x: (n_training_blocks, B, n)
            active_block: None (=all blocks) or bool of shape (n_blocks)
            chunk_s: int, block chunk starting index
            chunk_e: int, block chunk ending index

        Outputs:
            (n_training_blocks, B, n)
        """
        if training_blocks is None:
            return torch.sin(x*self.a[chunk_s:chunk_e])

        return torch.sin(x*self.a[training_blocks][chunk_s:chunk_e])


class GaussianActivation(nn.Module):
    def __init__(self, n_blocks, a=1):
        super().__init__()
        self.register_parameter('a',
            nn.Parameter(a*torch.ones(n_blocks, 1, 1)))

    def forward(self, x, training_blocks, chunk_s, chunk_e):
        """
        Inputs:
            x: (n_training_blocks, B, n)
            active_block: None (=all blocks) or bool of shape (n_blocks)
            chunk_s: int, block chunk starting index
            chunk_e: int, block chunk ending index

        Outputs:
            (n_training_blocks, B, n)
        """
        if training_blocks is None:
            return torch.exp(-x**2/(2*self.a[chunk_s:chunk_e]**2))

        return torch.exp(-x**2/(2*self.a[training_blocks][chunk_s:chunk_e]**2))
