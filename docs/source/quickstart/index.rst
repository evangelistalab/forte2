Quickstart
==========

To get started with Forte2, you can follow these steps:

1. Clone the Forte2 repository from GitHub:
    ```bash
    git clone git@github.com:evangelistalab/forte2.git
    cd forte2
    ```
2. Install the required dependencies in `environment.yml`.
   You can most easily do this in a new conda environment:
    ```bash
    conda env create -f environment.yml
    ```
3. Build the project:
    ```bash
    pip install --no-build-isolation -ve .
    ```
4. Run the quick tests:
    ```bash
    pytest -m "not slow"
    ```
