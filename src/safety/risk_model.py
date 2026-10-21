import torch
from torch import nn


class RiskPredictor(nn.Module):
    """
    Predict the probability that a safety violation will occur within a
    fixed future horizon after taking an action in the current state.

    The network itself returns logits rather than probabilities. Returning
    logits is numerically more stable when training with
    ``nn.BCEWithLogitsLoss``.

    Use ``predict_proba()`` when an actual probability in [0, 1] is needed
    during Safe-POR evaluation.

    Parameters
    ----------
    state_dim : int
        Number of state/observation features.

    action_dim : int
        Number of action features.

    hidden_dim : int, optional
        Number of neurons in each hidden layer.
        Default is 256.
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
        """
        Compute raw risk logits for state-action pairs.

        Parameters
        ----------
        state : torch.Tensor
            Tensor with shape (..., state_dim).

        action : torch.Tensor
            Tensor with shape (..., action_dim).

        Returns
        -------
        torch.Tensor
            Raw risk logits with shape (...,).

        Notes
        -----
        Do not apply sigmoid before ``BCEWithLogitsLoss`` during training.
        That loss function already performs the sigmoid operation internally.
        """

        # The risk model evaluates the complete state-action pair.
        x = torch.cat([state, action], dim=-1)

        # Squeeze the final dimension so a batch has shape (batch_size,)
        # rather than (batch_size, 1).
        return self.network(x).squeeze(-1)

    def predict_proba(self, state, action):
        """
        Convert the model's logits to violation probabilities.

        Parameters
        ----------
        state : torch.Tensor
            State tensor.

        action : torch.Tensor
            Action tensor.

        Returns
        -------
        torch.Tensor
            Estimated probability that a violation occurs within the
            K-step prediction horizon.

        Notes
        -----
        This method does not disable gradient tracking. During deployment,
        call it inside ``torch.no_grad()``.
        """

        logits = self.forward(state, action)

        return torch.sigmoid(logits)