import torch
import pytest
from src.models import ManualRNNCell, ManualRNNModel


# --- Fixtures ---
# A fixture is a reusable object that pytest creates fresh for each test.
# Instead of building a new cell inside every test function,
# we define it once here and pytest injects it automatically.

@pytest.fixture
def cell() -> ManualRNNCell:
    """A fresh ManualRNNCell with input_size=1, hidden_size=32."""
    return ManualRNNCell(input_size=1, hidden_size=32)


@pytest.fixture
def model() -> ManualRNNModel:
    """A fresh ManualRNNModel with input_size=1, hidden_size=32."""
    return ManualRNNModel(input_size=1, hidden_size=32)


# --- ManualRNNCell tests ---

def test_cell_weight_shapes(cell: ManualRNNCell):
    """W_xh, W_hh, and b must have the correct shapes."""
    assert cell.W_xh.shape == (1, 32),  f"Expected (1, 32), got {cell.W_xh.shape}"
    assert cell.W_hh.shape == (32, 32), f"Expected (32, 32), got {cell.W_hh.shape}"
    assert cell.b.shape    == (32,),    f"Expected (32,), got {cell.b.shape}"


def test_cell_output_shape(cell: ManualRNNCell):
    """One forward step must return hidden state shape (batch, hidden_size)."""
    batch_size  = 8
    x_t         = torch.randn(batch_size, 1)   # one timestep, one feature
    h_prev      = torch.zeros(batch_size, 32)  # initial hidden state
    h_t         = cell(x_t, h_prev)
    assert h_t.shape == (batch_size, 32), f"Expected (8, 32), got {h_t.shape}"


def test_cell_hidden_state_updates(cell: ManualRNNCell):
    """Hidden state must change after seeing a non-zero input."""
    x_t    = torch.ones(1, 1)    # non-zero input
    h_prev = torch.zeros(1, 32)  # start from zero
    h_t    = cell(x_t, h_prev)
    # If W_xh is non-zero (it is — initialized with randn),
    # the hidden state cannot remain all zeros after a non-zero input
    assert not torch.allclose(h_t, h_prev), \
        "Hidden state did not update — check W_xh initialization"


def test_cell_parameter_count(cell: ManualRNNCell):
    """Total parameters must be 1 + 32*32 + 32 = 1088."""
    total = sum(p.numel() for p in cell.parameters())
    assert total == 1088, f"Expected 1088 parameters, got {total}"


# --- ManualRNNModel tests ---

def test_model_output_shape(model: ManualRNNModel):
    """Model must return shape (batch, 1) for any valid input."""
    batch_size = 16
    seq_len    = 144
    x          = torch.randn(batch_size, seq_len, 1)
    out        = model(x)
    assert out.shape == (batch_size, 1), f"Expected (16, 1), got {out.shape}"


def test_model_output_is_scalar_per_sequence(model: ManualRNNModel):
    """Each sequence in the batch must produce exactly one forecast value."""
    x   = torch.randn(4, 144, 1)
    out = model(x)
    assert out.shape[0] == 4, "Batch dimension wrong"
    assert out.shape[1] == 1, "Output should be one value per sequence"


def test_model_different_inputs_different_outputs(model: ManualRNNModel):
    """Two different input sequences must produce different forecasts."""
    x1 = torch.randn(1, 144, 1)
    x2 = torch.randn(1, 144, 1)
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    assert not torch.allclose(out1, out2), \
        "Different inputs produced identical outputs — model may be degenerate"


def test_model_total_parameters(model: ManualRNNModel):
    """Total parameters must be 1088 (cell) + 33 (linear) = 1121."""
    total = sum(p.numel() for p in model.parameters())
    assert total == 1121, f"Expected 1121 parameters, got {total}"