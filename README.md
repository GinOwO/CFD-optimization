# High-Resolution CFD for Aerodynamic Optimization Using Dask, Coiled, and AWS

A parallel implementation of high-resolution computational fluid dynamics (CFD) for aerodynamic optimization using Dask, Coiled, and AWS. This project leverages OpenFOAM for CFD simulations and implements a differential evolution algorithm for airfoil shape optimization. In addition, it demonstrates how to build a custom Docker image and push it to Amazon ECR.

## Requirements

### Software
- Python 3.12.x or above
- GNU GCC 14.x or above
- OpenFOAM with parallel processing enabled
- OpenMPI
- Linux-Based OS (Kernel Version 4.x+)
- Docker (for building and pushing images)
- AWS CLI (for AWS operations)

### Python Dependencies
- SciPy
- NumPy
- Matplotlib
- Dask
- Coiled
- boto3 (optional, for additional AWS interactions)

### Cloud Resources
- AWS account with an ECR repository configured
- Dask cluster deployed via Coiled on AWS
- EC2 instances (as provisioned by Coiled)

## Installation

0. Install system dependencies as needed for your operating system.

1. Clone the repository:
    ```bash
    git clone https://github.com/ginowo/cfd-optimization.git
    cd cfd-optimization
    ```

2. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Build and upload a custom Docker image:
    1. **Prepare the custom image directory:**  
       Copy the base Dockerfile to a new directory and create a `Code` folder to hold the source:
       ```bash
       mkdir custom_image
       cp Dockerfile custom_image/
       mkdir custom_image/Code
       cp -r * custom_image/Code/
       ```
       *(Make sure to adjust the copy command to avoid duplicating the Dockerfile if necessary.)*
       
    2. **Build the Docker image:**  
       ```bash
       cd custom_image
       docker build -t <image-name> .
       ```
       
    3. **Tag and push the image to AWS ECR:**  
       Replace `<aws_account_id>`, `<region>`, and `your-ecr-repo` with your actual values:
       ```bash
       docker tag <image-name>:latest <aws_account_id>.dkr.ecr.<region>.amazonaws.com/your-ecr-repo:latest
       aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com
       docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/your-ecr-repo:latest
       ```

## Usage

1. **Set up AWS resources and Dask Cluster:**  
    - Ensure your custom Docker image is available in your ECR repository.
    - Create a Dask cluster using Coiled on AWS with your custom image. For example:
      ```python
      import coiled
      from dask.distributed import Client

      # Create a testing Coiled cluster using your custom Docker image from ECR
      # to ensure everything worked correctly, for example
      cluster = coiled.Cluster(
          name="my-cfd-opt-cluster",
          n_workers=3,
          worker_cpu=4,
          memory="8 GiB",
          docker_image="<aws_account_id>.dkr.ecr.<region>.amazonaws.com/your-ecr-repo:latest"
      )
      client = Client(cluster)
      print("Dask Cluster dashboard:", client.dashboard_link)
      ```

2. **Submit the Job:**  
    Run the job submission script in brute-force mode at the start which uses Dask and Coiled:
    ```bash
    python main.py
    ```
    Get the results from the scheduler nodes under `/opt/venv/results`
    Afterwards run in surrogate mode using the `-s` flag:
    ```bash
    python main.py -s
    ```

3. **Consolidate Results**:
    Get the results from the scheduler nodes under `/opt/venv/results` and store them for analysis.

4. **Analyze Results:**  
    Open and run the necessary cells in `analyze.ipynb` to visualize and review the optimization outcomes.

## Project Structure

- `main.py`: Core optimization logic
- `utils.py`: OpenFOAM interaction utilities
- `cst2coords.py`: Airfoil coordinate generation
- `foil_mesher.py`: Mesh generation
- `analyze.ipynb`: Result analysis and visualization

## License

This project is licensed under the GNU GPL v3 License – see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenFOAM Foundation for the CFD solver
- Coiled for simplifying Dask cluster management on AWS
- AWS for cloud computing infrastructure
- Pramudita Satria Palar for CST to Coords
- curiosityFluids for the foil mesher
- NeilsBongers for the base code
