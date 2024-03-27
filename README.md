# The Lighthouse Problem

 This project is a statistical approach to solving a classical problem derived from a Cambridge problem sheet by S. Gull and described in D. S. Sivia's "Data Analysis: A Bayesian Tutorial". This consisted of an implementation of Monte Carlo Markov Chain (MCMC) sampling techniques using Hamiltonian Monte Carlo (HMC) methods to estimate the unknown position of a lighthouse.

The repository contains a series of Python scripts that explore the problem. The following section provides instructions on how to run the `main.py` script.

by **Max Talberg**

## Running the script

### Setting Up the Project

1. **Clone the Repository:**
   - Clone the repository from GitLab:
     ```bash
     git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/S2_Assessment/mt942.git
     ```

2. **Set Up a Virtual Environment:**
   - Navigate to the project directory:
     ```bash
       cd src
       ```
   - Create a virtual environment:
     ```bash
     conda env create -f environment.yml
     ```
    - Activate the virtual environment:
      ```bash
      conda activate s2-env
      ```

### Running the Application

1. **Run the Command-Line Application:**
   - To solve a puzzle from the data file:
     ```bash
     python src/main.py
     ```

### Notes

- This setup assumes the use of `conda` over `pip`.

## Documentation

To build the documentation for the Sudoku Solver project, follow these steps:

### Docker
- When the Docker image for the Sudoku Solver is built, the project's Sphinx documentation is automatically built.

### Local
1. **Navigate to the `docs` directory:**
   - Navigate to the `docs` directory:
     ```bash
     cd docs
     ```
2. **Build the Documentation:**
    - Build the documentation:
      ```bash
      sphinx-build -b html source build
      ```
    - The documentation can be viewed by opening the `index.html` file in the `build` directory.

## Testing

To run tests for the Sudoku Solver project, follow these steps:

### Running Unit Tests
- Before running tests, ensure that the Python path is set up correctly to include the project's source code.


1. **Navigate to the project directory `mt942` and run Unit Tests:**

      ```bash
       pytest
     ```

### Notes
- If the unit tests cant find the `src` folder run:
    ```bash
      export PYTHONPATH=$PYTHONPATH:/src
    ```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Use of Generative Tools

This project has utilised GitHub Copilot for code completion and assistance with text prompts.

Although these tools aid the development process, its use was limited and only effective in scenarios where the developer had a clear idea of the code's structure and implementation. Other uses of these tools were used to aid the development of the project's documentation.

For instance:
- Plotting of functions
- Implementation of Latex and README
- Auto-complete of docstrings