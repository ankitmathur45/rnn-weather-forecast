import torch 
import torch.nn as nn

class ManualRNNCell(nn.Module):
    """
    A single RNN cell built from scratch using raw weight matrices.

    At each timestep t:
        h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b)

    Args:
        input_size:  number of features per timestep
        hidden_size: dimension of the hidden state vector
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        One timestep forward pass.

        Args:
            x_t:    current input,        shape (batch, input_size)
            h_prev: previous hidden state, shape (batch, hidden_size)

        Returns:
            h_t: new hidden state, shape (batch, hidden_size)
        """
        return torch.tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b)
    
class ManualRNNModel(nn.Module):
    """
    Full forecasting model built on top of ManualRNNCell.

    Architecture:
        Input (batch, seq_len, 1)
            -> ManualRNNCell at each timestep
            -> Final hidden state (batch, hidden_size)
            -> Linear layer
            -> Forecast (batch, 1)

    Args:
        input_size:  features per timestep
        hidden_size: dimension of hidden state
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = ManualRNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input sequences, shape (batch, seq_len, input_size)

        Returns:
            forecast: shape (batch, 1)
        """
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.hidden_size)
        for t in range(x.shape[1]):
            h = self.rnn_cell(x[:, t, :], h)
        return self.fc(h)