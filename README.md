# High-Resolution CFD for Aerodynamic Optimization Using Azure Batch

A parallel implementation of high-resolution computational fluid dynamics (CFD) for aerodynamic optimization using Azure Batch. This project leverages OpenFOAM for CFD simulations and implements a differential evolution algorithm for airfoil shape optimization.

## Features

- Distributed CFD optimization using Azure Batch
- OpenFOAM integration for flow simulation
- Class Shape Transformation (CST) airfoil parameterization
- Differential Evolution optimization algorithm
- Automated mesh generation and validation
- Result visualization and analysis

## Requirements

### Software
- Python 3.12.x or above
- GNU GCC 14.x or above
- OpenFOAM with parallel processing enabled
- OpenMPI
- Linux-Based OS (Kernel Version 4.x+)

### Python Dependencies
- SciPy
- NumPy
- Matplotlib
- Azure Batch SDK
- Azure Storage SDK

### Cloud Resources
- Azure Batch account
- Azure Storage account
- Standard_D2_v3 VMs (or equivalent)

## Installation

0. Install dependencies as mentioned above for your distribution/operating system

1. Clone the repository:
```
git clone https://github.com/ginowo/cfd-optimization.git
cd cfd-optimization
```

2. Install Python dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Set up Azure: 
    1. Create a Storage account, upload `install_deps.sh` to a container, `basic_template` to another container `tmpl`.
    2. Create a Batch account and a pool, specify nodes etc. and add the two files from the Storage containers as resource files and set `install_deps.sh` as start-up file.

2. Configure parameters in `.env`

3. Submit batch job:
```
python submit_job.py
```

4. Analyze results by running the necessary cells in `analyze.ipynb`

## Project Structure

- `submit_job.py`: Azure Batch job submission
- `main.py`: Core optimization logic
- `utils.py`: OpenFOAM interaction utilities
- `cst2coords.py`: Airfoil coordinate generation
- `foil_mesher.py`: Mesh generation
- `analyze.ipynb`: Result analysis and visualization

## License

This project is licensed under the GNU GPL v3 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenFOAM Foundation for the CFD solver
- Azure Batch team for cloud computing infrastructure
- Pramudita Satria Palar for CST to Coords
- curiosityFluids for foil mesher
- NeilsBongers for base code
