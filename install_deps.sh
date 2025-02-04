#!/bin/bash

# Update package list
sudo apt-get update -y

# Install Python 3 and pip
sudo apt-get install -y python3 python3-pip python3-dask python3-venv

# Install mpi (if not already installed)
sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev

# Install OpenFOAM

sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo add-apt-repository "http://dl.openfoam.org/ubuntu dev"
sudo apt-get update
sudo apt-get -y install openfoam-dev

# Clone the repository
cd /
mkdir -p cfd
rm -rf cfd
mkdir -p cfd
cd cfd
git clone https://github.com/GinOwO/CFD-optimization.git
cd CFD-optimization

# Create and activate a virtual environment
python3 -m venv venv
source ./venv/bin/activate

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

echo "Start task complete."