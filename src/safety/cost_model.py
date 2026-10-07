import torch
from torch import nn


class CostPredictor(nn.Module):
    """
    Predicts discounted future safety cost:

        Q_c(s, a)

    from the current state and action.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)

        return self.network(x).squeeze(-1)