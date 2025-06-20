# Physics-Informed Neural Networks (PINNs)

A Python implementation of Physics-Informed Neural Networks for solving differential equations and modeling physical systems.

## Features

- Solve Partial Differential Equations (PDEs)
- Solve Ordinary Differential Equations (ODEs)
- Handle both forward and inverse problems
- Customizable neural network architectures
- Built-in visualization tools
- Example problems with pre-configured settings

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pinn-project.git
cd pinn-project

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
pinn/
├── core/               # Core PINN implementation
│   ├── __init__.py
│   ├── network.py      # Neural network architectures
│   ├── losses.py       # Loss functions
│   └── trainer.py      # Training utilities
├── problems/           # Example problems
│   ├── __init__.py
│   ├── burgers.py      # Burgers' equation
│   ├── heat.py         # Heat equation
│   └── wave.py         # Wave equation
├── utils/              # Utility functions
│   ├── __init__.py
│   ├── visualize.py    # Visualization tools
│   └── data.py         # Data generation and processing
├── examples/           # Example scripts
│   ├── burgers_example.py
│   └── heat_example.py
├── tests/              # Unit tests
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Quick Start

```python
from pinn.core import PINN
from pinn.problems.burgers import BurgersProblem
import torch.optim as optim
import torch.nn as nn

# Define the problem
problem = BurgersProblem()

# Initialize the PINN
pinn = PINN(
    input_dim=2,  # (x, t)
    output_dim=1,  # u(x,t)
    hidden_layers=[20, 20, 20],
    activation=nn.Tanh()
)

# Set up optimizer
optimizer = optim.Adam(pinn.parameters(), lr=0.001)

# Train the model
pinn.train(problem, optimizer, epochs=1000, n_points=1000)

# Plot the results
problem.plot_solution(pinn)
```

## Examples

### 1D Heat Equation
```bash
python examples/heat_example.py
```

### 1D Burgers' Equation
```bash
python examples/burgers_example.py
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- SciPy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
