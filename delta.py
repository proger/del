__version__ = '0.1'

import torch
import torch.nn as nn

def shape(x):
    N, T, C = x.shape
    return N, T, C
    

class Delta(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gates = nn.Linear(dim, dim*3, bias=True)
        self.lr = nn.Linear(dim, 1, bias=True)
        self.output = nn.Linear(dim, dim, bias=True)
                
    def forward(self, x):
        q, k, v = self.gates(x).chunk(3, dim=-1)
        lr = self.lr(x)

        k = k / k.norm(dim=-1, keepdim=True)

        N, T, C = shape(x)
        K = torch.einsum('nsc,ntc->nst', k, k).tril_(diagonal=-1)
        u = v.clone()
        for t in range(1, T):
            u[:, t] = u[:, t].clone() - torch.einsum('nt,ntc->nc', K[:, t], u.clone())

        score = torch.einsum('nsk,ntk->nst', q, k)
        y = torch.einsum('nst,ntv->nsv', score, lr * u)
        return self.output(y)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.delta_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.delta = Delta(dim)

    def forward(self, x):
        x = x + self.delta(self.delta_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LM(nn.Module):
    def __init__(self, vocab_size=256, dim=32, num_layers=2):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, dim)
        self.backbone = nn.ModuleList([ResidualBlock(dim) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(dim)
        self.decoder = nn.Linear(dim, vocab_size, bias=False)

        with torch.no_grad():
            self.encoder.weight.normal_(std=dim**-0.5)
            self.decoder.weight.normal_(std=dim**-0.5)

        self.tie_weights_()

    def tie_weights_(self):
        self.decoder.weight = self.encoder.weight

    def forward(self, input_ids):
        N, T, *rest = input_ids.shape
        x = self.encoder(input_ids)
        if rest:
            x = x.sum(dim=tuple(range(2, 2+len(rest)))) # marginalize extra dimensions if present
        for block in self.backbone:
            x = block(x)
        x = self.output_norm(x)
        logits = self.decoder(x)
        return logits

    def parameter_groups(self, weight_decay=1e-2):
        return [
            {'params': self.encoder.parameters(), 'weight_decay': 0.0}, # decoder is tied here
            # do not decay biases and single-column parameters (forget_base, rmsnorm), those are usually scales
            {'params': (p for p in self.backbone.parameters() if p.dim() < 2 or getattr(p, '_no_weight_decay', False)), 'weight_decay': 0.0},
            {'params': (p for p in self.backbone.parameters() if p.dim() >= 2 and not getattr(p, '_no_weight_decay', False)), 'weight_decay': weight_decay},
            {'params': self.output_norm.parameters(), 'weight_decay': 0.0},
        ]


if __name__ == '__main__':
    vocab_size = 256
    model = LM(vocab_size=vocab_size)
    
    print(model(torch.randint(vocab_size, (3, 9))))