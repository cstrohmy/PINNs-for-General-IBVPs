# PINNs-for-General-IBVPs
Allows user to specify an essentially arbitrary initial boundary-value problem (IBVP) and train a physics-informed neural network (PINN) to solve said problem.


## Overview

This project aims to solve Partial Differential Equations (PDEs) using neural networks. It is particularly focused on boundary value problems like the Dirichlet problem on geometric domains. The framework is modular, designed to be highly customizable for solving a wide range of PDEs defined on complex domains.

## Modules

### `domain.py`

- Defines geometric domains on which the PDE is solved. 
- Supports both interior and boundary pieces.

### `ibvp.py`

- Stands for Initial-Boundary Value Problem.
- Specifies the PDEs, dependent variables, and additional data like fill-ins for boundary conditions.

### `algorithm.py`

- Contains the learning algorithm for training neural networks to approximate solutions to PDEs.
- Takes into consideration different sampling strategies, loss functions, and domain-specific pieces.

### `demo.py`

- Provides a concrete example by solving the Laplace equation on a unit disk.
- Utilizes all the above modules to define the problem and train a neural network for solving it.

## Usage

1. Define your geometric domain using `InteriorPiece` and `BoundaryPiece` in `domain.py`.
2. Specify the PDEs, initial or boundary conditions in `ibvp.py`.
3. Train a neural network model using `LearningAlgorithm` in `algorithm.py`.

## Additional Information

- For a complete example that solves a Laplace equation on a unit disk with specific boundary conditions, please refer to `demo.py`.
- The architecture and size of the neural network model can be easily adapted in the `demo.py` file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues to discuss improvements or feature requests.

## Contact

For questions or clarifications, feel free to contact the maintainer.

