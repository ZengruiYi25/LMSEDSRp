import torch
import torch.nn as nn


class param_Block(nn.Module):
    def __init__(self, latent, num_variables, dropout, mult=4):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_variables*3, latent),
            nn.GroupNorm(1, latent),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(latent, latent*mult),
            nn.GroupNorm(1, latent*mult),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(latent*mult, latent),
        )

        self.res = nn.Linear(num_variables*3, latent)

        self.norm = nn.Sequential(
            nn.GroupNorm(1, latent),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        self.out = nn.Linear(latent, 3)

    def forward(self, x, M, B):
        x = torch.concatenate((x, M, B), dim=-1)

        x = self.fc(x) + self.res(x)
        x = self.norm(x)

        coefs = self.out(x)
        return coefs


class M_Emb_Block(nn.Module):
    def __init__(self, latent, num_variables, dropout):
        super().__init__()

        self.emb = nn.Sequential(
            nn.Embedding(10, num_variables),
        )

    def forward(self, m):
        return self.emb(m)


class B_Emb_Block(nn.Module):
    def __init__(self, latent, num_variables, dropout):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(1024, num_variables),
        )

    def forward(self, b):
        return self.fc(b)


class Net(nn.Module):
    def __init__(self, latent, num_variables=7, dropout=0.2):
        super().__init__()
        self.num_variables = num_variables

        self.M_emb = M_Emb_Block(latent, num_variables, dropout)

        self.B_emb = B_Emb_Block(latent, num_variables, dropout)

        self.param = param_Block(latent, num_variables, dropout)

    def forward(self, x):
        """
        x shape: [batch, []]
        [0] --> M: Material type
        [1] --> F: Frequency
        [2] --> f_sin: Corrected frequency
        [3] --> Bm: Peak magnetic flux density
        [4] --> C: Temperature
        [5] --> Hdc: DC bias
        [6, 7] --> D: Duty cycle
        [8:] --> B: Magnetic flux density
        """
        n = self.num_variables + 1

        M = self.M_emb(x[:, 0].type(torch.int32))
        B = self.B_emb(x[:, n:])

        coefs = self.param(x[:, 1:n], M, B)

        P = coefs[:, 0] + coefs[:, 1] * x[:, 1] + coefs[:, 2] * x[:, 3]

        return coefs, P