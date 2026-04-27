# RNN Weather Forecast
Predicts next-step temperature from the JENA climate dataset using three progressively powerful sequence models built in PyTorch

## What this teaches
- How a RNN cell works at the weight-matrix level
- Backpropagation through time (BPTT) and vanishing gradients
- Why LSTMs exist and what the cell state actually does

## Architecture
RAW CSV -> Normalize -> Sliding window sequence -> RNN/LSTM -> Forecast

## Models built
1. Manual RNN cell - weight matrices written by hand
2. nn.RNN - same math, PyTorch abstraction
3. nn.LSTM - cell state, three gates, vanishing gradient fix

## Setup
```bash
uv venv .vevn --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt
```

## Tech stack
Python 3.12     PyTorch     Pandas      Plotly      Pytest

